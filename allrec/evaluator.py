import torch

class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }
        self.maxk = max(config['topk'])
    def calculate_pos_index(self, preds, labels):
            preds = preds.detach()
            labels = labels.detach()
            if labels.dim() !=2:
                raise ValueError(f"labels must be a 1D or 2D tensor, but got {labels.shape}")
            pos_index = (preds == labels)
            return pos_index
    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).float()
    def ndcg_at_k(self, pos_index, k):
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).float()
    def calculate_metrics(self, preds, labels):
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](
                    pos_index, k)
        return results
