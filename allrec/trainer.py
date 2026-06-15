import os
import sys
import json
from datetime import datetime
from collections import defaultdict, OrderedDict
from logging import getLogger
from typing import Union
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from accelerate import Accelerator
from transformers.optimization import get_scheduler
from tqdm import tqdm
from swanlab import init as swanlab_init
from swanlab import log as swanlab_log
from allrec.dataset import AbstractDataset
from allrec.model import AbstractModel
from allrec.tokenizer import AbstractTokenizer
from allrec.evaluator import Evaluator
from allrec.utils import (get_file_name, log, Colors)

class Trainer:
    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.accelerator: Accelerator = config['accelerator']
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()
        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            self.config['model'],
            get_file_name(self.config, suffix='.pth')
        )
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)
    def fit(self, train_dataloader, val_dataloader):
        total_n_steps = len(train_dataloader) * self.config['epochs']
        warmup_steps = int(self.config['warmup_rate'] * total_n_steps)
        n_epochs = np.ceil(total_n_steps / len(train_dataloader)).astype(int)
        self.log(f'[Trainer] Total training steps per device: {total_n_steps}, warmup steps: {warmup_steps}')
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_n_steps,
        )
        self.model, optimizer, scheduler, train_dataloader, val_dataloader = self.accelerator.prepare(
            self.model, optimizer, scheduler, train_dataloader, val_dataloader
        )
        print(f"[Trainer] Epochs: {n_epochs}, rank: {self.accelerator.process_index}, num processes: {self.accelerator.num_processes}, train batches: {len(train_dataloader)}")
        self.log(f"[Trainer] Epochs: {n_epochs}, rank: {self.accelerator.process_index}, num processes: {self.accelerator.num_processes}, train batches: {len(train_dataloader)}")
        best_epoch = 0
        best_val_score = -1
        for epoch in range(n_epochs):
            train_start_time = time.time()
            self.model.train()
            total_loss = torch.tensor(0.0, device = self.accelerator.device)
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"{Colors.CYAN}Training{Colors.RESET} {Colors.YELLOW}[Epoch {epoch + 1:>3}{Colors.RESET}]",
                disable=not self.accelerator.is_main_process,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{postfix}]"
            )
            for batch_idx, batch in enumerate(train_progress_bar):
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss += loss.detach()
            local_avg_loss = total_loss / len(train_dataloader)
            avg_loss = self.accelerator.reduce(local_avg_loss, reduction="mean").item()
            if self.config['use_swanlab'] and self.accelerator.is_main_process:
                swanlab_log(
                    {
                        "Loss/epoch_loss": avg_loss,
                        "lr/epoch": scheduler.get_last_lr()[0]
                    },
                    step=epoch + 1
                )
            msg = f'[Trainer][Epoch {epoch + 1}] Train loss: {avg_loss:.8f}'
            train_end_time = time.time()
            train_time = train_end_time - train_start_time
            msg += f', train time: {train_time:.2f}s'
            self.log(msg)
            if (epoch + 1) % self.config['eval_interval'] == 0:
                eval_start_time = time.time()
                all_results = self.evaluate(val_dataloader, split='val')
                eval_end_time = time.time()
                eval_time = eval_end_time - eval_start_time
                if self.accelerator.is_main_process:
                    for key in all_results:
                        if self.config['use_swanlab']:
                            swanlab_log(
                                {f"Val_Metric/{key}": all_results[key]}, step=epoch + 1
                            )
                        print(f'[Trainer] {Colors.GREEN}{key}: {Colors.MAGENTA}{all_results[key]:.5f}{Colors.RESET} ', end="")
                        if key == list(all_results.keys())[-1]:
                            print()
                    self.log(f'[Trainer][Epoch {epoch + 1}] Validation results: {all_results}, validation time: {eval_time:.2f}s')
                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        unwrapped_model = self.accelerator.unwrap_model(
                            self.model)
                        torch.save(unwrapped_model.state_dict(),
                                    self.saved_model_ckpt)
                        self.log(f'[Trainer][Epoch {epoch + 1}] Saved model checkpoint: {self.saved_model_ckpt}')
                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    self.log(f'[Trainer] Early stopping at epoch {epoch + 1}')
                    break
        self.log(f'[Trainer] Best epoch: {best_epoch}, best validation score: {best_val_score}')
        return best_epoch, best_val_score
    def evaluate(self, dataloader, split='test'):
        self.model.eval()
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"{Colors.CYAN}Eval{Colors.RESET} - {Colors.YELLOW}{split}{Colors.RESET}",
            disable=not self.accelerator.is_main_process,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
        )
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        all_preds_list = []
        all_labels_list = []
        for batch in val_progress_bar:
            with torch.no_grad():
                preds = unwrapped_model.generate(
                    batch,
                    n_return_sequences=self.evaluator.maxk
                )
                all_preds_list.append(preds)
                all_labels_list.append(batch['labels'])
        all_preds = torch.cat(all_preds_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)
        del all_preds_list, all_labels_list
        global_preds, global_labels = self.accelerator.gather_for_metrics((all_preds, all_labels))
        results = self.evaluator.calculate_metrics(global_preds, global_labels)
        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                if key in results:
                    output_results[key] = results[key].detach().mean().item()
                else:
                    output_results[key] = 0.0
        self.accelerator.wait_for_everyone()
        return output_results
    def end(self):
        self.accelerator.end_training()
    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
