from datetime import datetime, timedelta
import os
from logging import getLogger
from typing import Union
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from swanlab import init as swanlab_init
from allrec.dataset import AbstractDataset
from allrec.model import AbstractModel
from allrec.tokenizer import AbstractTokenizer
from allrec.utils import (
    get_config, init_seed, init_logger, init_device,
    get_dataset, get_tokenizer, get_model, get_trainer, log, Colors,
    maybe_log_custom_pipeline,
)

class Pipeline:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        category: str,
        checkpoint_path: str = None,
        config_dict: dict = None,
    ):
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_dict=config_dict
        )
        self.config['model'] = model_name
        self.config['dataset'] = dataset_name
        self.config['category'] = category
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        kwargs_handlers = None
        if world_size > 1:
            kwargs_handlers = [
                DistributedDataParallelKwargs(
                    find_unused_parameters=bool(self.config.get("ddp_find_unused_parameters", False))
                )
            ]
        self.accelerator = Accelerator(kwargs_handlers=kwargs_handlers)
        self.config['accelerator'] = self.accelerator
        self.config['device'], self.config['use_ddp'] = init_device()
        self.checkpoint_path = checkpoint_path
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        maybe_log_custom_pipeline(type(self), self.config)
        self.log_config(self.config)
        with self.accelerator.main_process_first():
            if self.config['use_swanlab'] and self.accelerator.is_main_process:
                swanlab_init(
                    project = "AllRec",
                    workspace = "A611699",
                    config = self.config,
                )
        with self.accelerator.main_process_first():
            self.raw_dataset = get_dataset(dataset_name, model_name=model_name)(self.config)
            self.log(self.raw_dataset)
            self.split_datasets = self.raw_dataset.split()
        with self.accelerator.main_process_first():
            self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)
        with self.accelerator.main_process_first():
            for k, v in self.split_datasets.items():
                self.log("split datasets type:{}, len:{}".format(k, len(v)))
            self.tokenized_datasets = self.tokenizer.tokenize_datasets(self.split_datasets)
        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.raw_dataset, self.tokenizer)
            if checkpoint_path is not None:
                self.model.load_state_dict(
                    torch.load(checkpoint_path,
                               map_location=self.config['device'])
                )
                self.log(f'Loaded model checkpoint from {checkpoint_path}')
            self.log(self.model)
            self.log(self.model.n_parameters)
        self.trainer = get_trainer(model_name)(self.config, self.model, self.tokenizer)
    def run(self):
        train_dataloader = DataLoader(
            self.tokenized_datasets['train'],
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            collate_fn=self.tokenizer.collate_fn['train'],
            num_workers=self.config['data_loader_num_workers'],
            pin_memory=self.config['pin_memory']
        )
        self.log(f'Train dataloader has record num: {len(self.tokenized_datasets["train"])}')
        val_dataloader = DataLoader(
            self.tokenized_datasets['val'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['val'],
            num_workers=self.config['data_loader_num_workers'],
            pin_memory=self.config['pin_memory']
        )
        self.log(f'Validation dataloader has record num: {len(self.tokenized_datasets["val"])}')
        test_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test'],
            num_workers=self.config['data_loader_num_workers'],
            pin_memory=self.config['pin_memory']
        )
        self.log(f'Test dataloader has record num: {len(self.tokenized_datasets["test"])}')
        best_epoch, best_val_score = self.trainer.fit(train_dataloader, val_dataloader)
        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)
        if self.checkpoint_path is None:
            self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt,weights_only=False))
        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )
        if self.accelerator.is_main_process and self.checkpoint_path is None:
            self.log(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')
        import time
        start_time = time.time()
        test_results = self.trainer.evaluate(test_dataloader)
        end_time = time.time()
        time_consumed = end_time - start_time
        self.log(f'Test time consumed: {time_consumed:.2f} seconds')
        if self.accelerator.is_main_process:
            for key in test_results:
                print(f'{Colors.GREEN}{key}: {Colors.MAGENTA}{test_results[key]:.5f}{Colors.RESET}', end="")
                if key == list(test_results.keys())[-1]:
                    print()
            self.log(f'Test Results: {test_results}')
            self.log(f"时间日期：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"时间日期：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.trainer.end()
        return {
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'test_results': test_results,
        }
    def log(self, message, level='info'):
        return log(
            message,
            self.config['accelerator'],
            self.logger,
            level=level
        )
    def log_config(self, config: dict):
        for k, v in sorted(config.items(), key=lambda item: item[0]):
            self.log(f"  {k}: {v} ({type(v).__name__})")
