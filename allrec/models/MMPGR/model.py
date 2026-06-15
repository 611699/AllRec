from logging import getLogger
import math
from typing import Optional, List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from allrec.dataset import AbstractDataset
from allrec.model import AbstractModel
from allrec.tokenizer import AbstractTokenizer
from dataclasses import dataclass
from transformers.utils import ModelOutput

@dataclass

class CustomGPT2BlockOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    attn_scores: Optional[torch.FloatTensor] = None
    final_states: Optional[torch.FloatTensor] = None
    modal_fusion_final_states: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    loss_uniformity: Optional[torch.FloatTensor] = None
    loss_similarity_mse: Optional[torch.FloatTensor] = None
    v: Optional[torch.FloatTensor] = None

class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.activation_fun = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, x):
        return x+self.activation_fun(self.linear(self.norm(x)))

class CustomGPT2Block(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.hidden_size = config["n_embd"]
        self.num_heads = config["n_head"]
        self.head_dim = self.hidden_size//self.num_heads
        assert self.head_dim * \
            self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"
        self.ln_1 = nn.LayerNorm(
            self.hidden_size, eps=config['layer_norm_epsilon'], elementwise_affine=True)
        self.c_attn_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_attn_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_attn_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = nn.Dropout(config['attn_pdrop'], inplace=False)
        self.resid_dropout = nn.Dropout(config['resid_pdrop'], inplace=False)
        self.ln_2 = nn.LayerNorm(
            self.hidden_size, eps=config['layer_norm_epsilon'], elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, config['n_inner']),
            nn.GELU(),
            nn.Linear(config['n_inner'], self.hidden_size),
            nn.Dropout(config['resid_pdrop'], inplace=False)
        )
        for module in [self.c_attn_q, self.c_attn_k, self.c_attn_v, self.c_proj, self.mlp[0], self.mlp[2]]:
            nn.init.normal_(module.weight, mean=0.0,
                            std=config['initializer_range'])
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, nH, L, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, nH*D)
    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, nH, L, D = q.size()
        attn_scores = torch.matmul(
            q, k.transpose(-1, -2))/math.sqrt(self.head_dim)
        if not hasattr(self, "causal_mask"):
            mask = torch.tril(torch.ones((L, L), device=q.device, dtype=torch.bool)).unsqueeze(
                0).unsqueeze(0)
            self.register_buffer("causal_mask", mask)
        padding_mask = attention_mask.bool().unsqueeze(1).unsqueeze(
            2).expand(-1, nH, L, -1)
        combined_mask = self.causal_mask & padding_mask
        attn_scores = attn_scores.masked_fill(
            ~combined_mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        out = torch.matmul(attn_probs, v)
        return out, attn_scores
    def forward(self, q_input: torch.Tensor, k_input: torch.Tensor, v_input: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        residual = v_input
        q_norm = self.ln_1(q_input)
        k_norm = self.ln_1(k_input)
        v_norm = self.ln_1(v_input)
        q = self._split_heads(self.c_attn_q(q_norm))
        k = self._split_heads(self.c_attn_k(k_norm))
        v = self._split_heads(self.c_attn_v(v_norm))
        attn_output, attn_scores = self._attention(q, k, v, attention_mask)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        x = residual + attn_output
        x = self.resid_dropout(x)
        x = x + self.mlp(self.ln_2(x))
        outputs = CustomGPT2BlockOutput(
            last_hidden_state=x,
            attn_scores=attn_scores,
            v=v,
        )
        return outputs

class CrossAttentionGPT2Block(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.hidden_size = config["n_embd"]
        self.num_heads = config["n_head"]
        self.head_dim = self.hidden_size//self.num_heads
        assert self.head_dim * \
            self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = nn.Dropout(config['attn_pdrop'], inplace=False)
        self.resid_dropout = nn.Dropout(config['resid_pdrop'], inplace=False)
        self.ln_1 = nn.LayerNorm(
            self.hidden_size, eps=config['layer_norm_epsilon'], elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, config['n_inner']),
            nn.GELU(),
            nn.Linear(config['n_inner'], self.hidden_size),
            nn.Dropout(config['resid_pdrop'], inplace=False)
        )
        for module in [self.c_proj, self.mlp[0], self.mlp[2]]:
            nn.init.normal_(module.weight, mean=0.0,
                            std=config['initializer_range'])
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    def forward(self, v: torch.Tensor, attention_mask: torch.Tensor, attention_scores: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        B, nH, L, D = v.size()
        residual = v
        attn_probs = F.softmax(attention_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        out = torch.matmul(attn_probs, v)
        out = self._merge_heads(out)
        out = self.c_proj(out)
        residual = self._merge_heads(residual)
        out = residual + out
        out = self.resid_dropout(out)
        out = out + self.mlp(self.ln_1(out))
        outputs = CustomGPT2BlockOutput(
            last_hidden_state=out,
        )
        return outputs
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, nH, L, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, nH*D)

class MMPGR(AbstractModel):
    def __init__(self, config: dict, dataset: AbstractDataset, tokenizer: AbstractTokenizer):
        super(MMPGR, self).__init__(config, dataset, tokenizer)
        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])
        self.logger = getLogger()
        self.model_name = 'MMPGR'
        self.log(f"[Model] Initializing MMPGR.")
        self.log(
            f'[Model] item_id2tokens mapping shape: {self.item_id2tokens.shape}')
        self.config = config
        self.text_item_embs = torch.from_numpy(tokenizer.text_embs).to(torch.float32).to(self.config['device'])
        memory_bytes = self.text_item_embs.nelement() * self.text_item_embs.element_size()
        memory_mb = memory_bytes / 1024 / 1024
        self.log(f"[Model] Cached text item embeddings on GPU: shape={self.text_item_embs.shape}, memory={memory_mb:.2f} MB")
        self.image_item_embs = torch.from_numpy(tokenizer.image_embs).to(torch.float32).to(self.config['device'])
        memory_bytes = self.image_item_embs.nelement() * self.image_item_embs.element_size()
        memory_mb = memory_bytes / 1024 / 1024
        self.log(f"[Model] Cached image item embeddings on GPU: shape={self.image_item_embs.shape}, memory={memory_mb:.2f} MB")
        self.text_similarity_min = float(tokenizer.text_similarity_min)
        self.text_similarity_max = float(tokenizer.text_similarity_max)
        self.image_similarity_min = float(tokenizer.image_similarity_min)
        self.image_similarity_max = float(tokenizer.image_similarity_max)
        self.token_embs = nn.Embedding(
            tokenizer.vocab_size, config["n_embd"],padding_idx=0).to(self.config['device'])
        self.position_embs = nn.Embedding(
            tokenizer.max_history_sequence_len, config["n_embd"]).to(self.config['device'])
        for module in [self.token_embs, self.position_embs]:
            nn.init.normal_(module.weight, mean=0.0,
                            std=config['initializer_range'])
        self.ln_f_text = nn.LayerNorm(
            config['n_embd'], eps=config['layer_norm_epsilon'], elementwise_affine=True)
        self.ln_f_image = nn.LayerNorm(
            config['n_embd'], eps=config['layer_norm_epsilon'], elementwise_affine=True)
        self.ln_f_image_to_text = nn.LayerNorm(
            config['n_embd'], eps=config['layer_norm_epsilon'], elementwise_affine=True)
        self.ln_f_text_to_image = nn.LayerNorm(
            config['n_embd'], eps=config['layer_norm_epsilon'], elementwise_affine=True)
        self.embd_dropout = nn.Dropout(config['embd_pdrop'], inplace=False)
        gpt2blockconfig = {
            "n_embd": config["n_embd"],
            "n_head": config["n_head"],
            "n_inner": config["n_inner"],
            "activation_function": config["activation_function"],
            "resid_pdrop": config["resid_pdrop"],
            "attn_pdrop": config["attn_pdrop"],
            "layer_norm_epsilon": config["layer_norm_epsilon"],
            "initializer_range": config["initializer_range"]
        }
        self.GPT2BlockForText = CustomGPT2Block(gpt2blockconfig)
        self.GPT2BlockForImage = CustomGPT2Block(gpt2blockconfig)
        self.GPT2BlockImageToText = CrossAttentionGPT2Block(gpt2blockconfig)
        self.GPT2BlockTextToImage = CrossAttentionGPT2Block(gpt2blockconfig)
        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list_for_text = []
        pred_head_list_for_image = []
        for i in range(self.n_pred_head//2):
            pred_head_list_for_text.append(ResBlock(self.config['n_embd']))
            pred_head_list_for_image.append(ResBlock(self.config['n_embd']))
        self.pred_heads_for_text = nn.ModuleList(pred_head_list_for_text)
        self.pred_heads_for_image = nn.ModuleList(pred_head_list_for_image)
        self.temperature = config["temperature"]
        self.loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.ignored_label
        )
    @property
    def n_parameters(self) -> str:
        total_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        emb_params = sum(
            p.numel() for p in self.token_embs.parameters() if p.requires_grad
        ) + sum(
            p.numel() for p in self.position_embs.parameters() if p.requires_grad
        )
        non_emb_params = total_params - emb_params
        max_width = len(str(total_params))
        summary = (
            f'#Embedding parameters:     {emb_params:>{max_width}}\n'
            f'#Non-embedding parameters: {non_emb_params:>{max_width}}\n'
            f'#Total trainable parameters:{total_params:>{max_width}}\n'
        )
        return summary
    def _map_item_tokens(self) -> torch.Tensor:
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(
                self.tokenizer.item2tokens[item])
        return item_id2tokens
    def log(self, message, level='info'):
        from allrec.utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)
    def _compute_realtime_similarity_scores(
        self,
        item_embs: torch.Tensor,
        target_item_ids: torch.Tensor,
        similarity_min: float,
        similarity_max: float,
    ) -> torch.Tensor:
        with torch.no_grad():
            target_embs = item_embs[target_item_ids]
            similarity_scores = torch.matmul(target_embs, target_embs.T)
            return 2 * (similarity_scores - similarity_min) / (similarity_max - similarity_min + 1e-12) - 1
    def forward(self, batch: dict, return_loss=True) -> CustomGPT2BlockOutput:
        input_tokens = self.item_id2tokens[batch['input_ids']]
        text_input_tokens = input_tokens[:, :, :self.tokenizer.n_digit//2]
        image_input_tokens = input_tokens[:, :, self.tokenizer.n_digit//2:]
        text_input_embs = self.token_embs(text_input_tokens).mean(
            dim=-2)
        image_input_embs = self.token_embs(image_input_tokens).mean(
            dim=-2)
        position_embs = self.position_embs(torch.arange(text_input_embs.shape[1], device=self.config['device']))
        text_input_embs = text_input_embs + position_embs.unsqueeze(0).expand(text_input_embs.shape[0], -1, -1)
        image_input_embs = image_input_embs + position_embs.unsqueeze(0).expand(image_input_embs.shape[0], -1, -1)
        text_input_embs = self.embd_dropout(text_input_embs)
        image_input_embs = self.embd_dropout(image_input_embs)
        text_outputs = self.GPT2BlockForText(
            q_input=text_input_embs,
            k_input=text_input_embs,
            v_input=text_input_embs,
            attention_mask=batch['attention_mask']
        )
        text_outputs.last_hidden_state = self.ln_f_text(text_outputs.last_hidden_state)
        image_outputs = self.GPT2BlockForImage(
            q_input=image_input_embs,
            k_input=image_input_embs,
            v_input=image_input_embs,
            attention_mask=batch['attention_mask']
        )
        image_outputs.last_hidden_state = self.ln_f_image(image_outputs.last_hidden_state)
        image_to_text_outputs = self.GPT2BlockImageToText(
            v=text_outputs.v,
            attention_mask=batch['attention_mask'],
            attention_scores=image_outputs.attn_scores,
        )
        image_to_text_outputs.last_hidden_state = self.ln_f_image_to_text(image_to_text_outputs.last_hidden_state)
        text_to_image_outputs = self.GPT2BlockTextToImage(
            v=image_outputs.v,
            attention_mask=batch['attention_mask'],
            attention_scores=text_outputs.attn_scores,
        )
        text_to_image_outputs.last_hidden_state = self.ln_f_text_to_image(text_to_image_outputs.last_hidden_state)
        if return_loss:
            label_mask = batch['labels'].view(-1) != -100
            text_outputs.last_hidden_state = text_outputs.last_hidden_state.view(-1, self.config['n_embd'])[label_mask]
            image_outputs.last_hidden_state = image_outputs.last_hidden_state.view(-1, self.config['n_embd'])[label_mask]
            image_to_text_outputs.last_hidden_state = image_to_text_outputs.last_hidden_state.view(-1, self.config['n_embd'])[label_mask]
            text_to_image_outputs.last_hidden_state = text_to_image_outputs.last_hidden_state.view(-1, self.config['n_embd'])[label_mask]
        if not return_loss:
            text_outputs.last_hidden_state = text_outputs.last_hidden_state.gather(
                dim = 1,
                index = (batch['seq_lens']-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.config['n_embd'])
            ).squeeze(dim=1)
            image_outputs.last_hidden_state = image_outputs.last_hidden_state.gather(
                dim = 1,
                index = (batch['seq_lens']-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.config['n_embd'])
            ).squeeze(dim=1)
            image_to_text_outputs.last_hidden_state = image_to_text_outputs.last_hidden_state.gather(
                dim = 1,
                index = (batch['seq_lens']-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.config['n_embd'])
                .expand(-1, 1, self.config['n_embd'])
            ).squeeze(dim=1)
            text_to_image_outputs.last_hidden_state = text_to_image_outputs.last_hidden_state.gather(
                dim = 1,
                index = (batch['seq_lens']-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.config['n_embd'])
            ).squeeze(dim=1)
        text_final_states = [
            self.pred_heads_for_text[i](
                text_outputs.last_hidden_state).unsqueeze(-2)
            for i in range(self.n_pred_head//2)
        ]
        image_final_states = [
            self.pred_heads_for_image[i](
                image_outputs.last_hidden_state).unsqueeze(-2)
            for i in range(self.n_pred_head//2)
        ]
        final_states = torch.cat(text_final_states+image_final_states, dim=-2)
        outputs = CustomGPT2BlockOutput()
        outputs.final_states = final_states
        image_to_text_final_states = [
            self.pred_heads_for_text[i](image_to_text_outputs.last_hidden_state).unsqueeze(-2)
            for i in range(self.n_pred_head//2)
        ]
        text_to_image_final_states = [
            self.pred_heads_for_image[i](text_to_image_outputs.last_hidden_state).unsqueeze(-2)
            for i in range(self.n_pred_head//2)
        ]
        modal_fusion_final_states = torch.cat(
            image_to_text_final_states+text_to_image_final_states, dim=-2)
        outputs.modal_fusion_final_states = modal_fusion_final_states
        if return_loss:
            assert 'labels' in batch, "The batch must contain the labels."
            token_emb = self.token_embs.weight[1:-1]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
            token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
            target_item_ids = batch['labels'].view(-1)[label_mask]-1
            assert torch.all(target_item_ids >= 0) and torch.all(target_item_ids < self.dataset.n_items-1), "target_item_ids must be in 0 to n_items-1"
            target_text_token_labels = token_labels[:, :self.tokenizer.n_digit//2]
            target_image_token_labels = token_labels[:, self.tokenizer.n_digit//2:]
            target_item_text_embs = self.token_embs(target_text_token_labels).mean(dim=-2)
            target_item_text_embs = F.normalize(target_item_text_embs, dim=-1)
            target_item_image_embs = self.token_embs(target_image_token_labels).mean(dim=-2)
            target_item_image_embs = F.normalize(target_item_image_embs, dim=-1)
            target_text_similarity_scores = torch.matmul(target_item_text_embs, target_item_text_embs.T)
            target_image_similarity_scores = torch.matmul(target_item_image_embs, target_item_image_embs.T)
            to_text_similarity_scores = self._compute_realtime_similarity_scores(
                self.text_item_embs,
                target_item_ids,
                self.text_similarity_min,
                self.text_similarity_max,
            )
            to_image_similarity_scores = self._compute_realtime_similarity_scores(
                self.image_item_embs,
                target_item_ids,
                self.image_similarity_min,
                self.image_similarity_max,
            )
            text_mse_loss = F.mse_loss(target_text_similarity_scores, to_text_similarity_scores)
            image_mse_loss = F.mse_loss(target_image_similarity_scores, to_image_similarity_scores)
            outputs.similarity_mse_loss = text_mse_loss + image_mse_loss
            single_modal_selected_states = final_states
            single_modal_selected_states = F.normalize(single_modal_selected_states, dim=-1)
            single_modal_selected_states = torch.chunk(single_modal_selected_states, self.n_pred_head, dim=1)
            single_modal_token_logits = [
                torch.matmul(
                    single_modal_selected_states[i].squeeze(dim=1),
                    token_embs[i].T
                )/self.temperature
                for i in range(self.n_pred_head)
            ]
            single_modal_losses = [
                self.loss_fct(
                    single_modal_token_logits[i],
                    token_labels[:, i]-i*self.config['codebook_size']-1
                )
                for i in range(self.n_pred_head)
            ]
            modal_fusion_selected_states = modal_fusion_final_states
            modal_fusion_selected_states = F.normalize(modal_fusion_selected_states, dim=-1)
            modal_fusion_selected_states = torch.chunk(modal_fusion_selected_states, self.n_pred_head, dim=1)
            modal_fusion_token_logits = [
                torch.matmul(
                    modal_fusion_selected_states[i].squeeze(
                        dim=1),
                    token_embs[i].T
                )/self.temperature
                for i in range(self.n_pred_head)
            ]
            modal_fusion_losses = [
                self.loss_fct(
                    modal_fusion_token_logits[i],
                    token_labels[:, i]-i*self.config['codebook_size']-1
                )
                for i in range(self.n_pred_head)
            ]
            outputs.loss = torch.mean(torch.stack(single_modal_losses+modal_fusion_losses))
            outputs.loss_ce = None
            outputs.loss_uniformity = None
            outputs.loss += outputs.similarity_mse_loss*self.config['similarity_mse_loss_weight']
            outputs.loss_similarity_mse = outputs.similarity_mse_loss*self.config['similarity_mse_loss_weight']
        return outputs
    def generate(self, batch: dict, n_return_sequences: int = 1) -> torch.Tensor:
        token_emb = self.token_embs.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        outputs = self.forward(batch, return_loss=False)
        states = outputs.final_states
        states = F.normalize(states, dim=-1)
        modal_fusion_states = outputs.modal_fusion_final_states
        modal_fusion_states = F.normalize(modal_fusion_states, dim=-1)
        states = states+modal_fusion_states
        logits = [
            torch.matmul(
                states[:,i, :],
                token_embs[i].T
            )/self.temperature
            for i in range(self.n_pred_head)
        ]
        token_logits = torch.cat(logits, dim=-1)
        item_logits = torch.gather(
            input=token_logits.unsqueeze(-2).expand(-1,self.dataset.n_items-1, -1),
            dim=-1,
            index=(self.item_id2tokens[1:, :]-1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)
        ).mean(dim=-1)
        preds = item_logits.topk(n_return_sequences, dim=-1).indices+1
        return preds
