from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import numpy as np
import json
import requests
from sklearn.decomposition import NMF
import torch
from tqdm import tqdm
import os
from logging import getLogger
from allrec.utils import log
from allrec.dataset import AbstractDataset
from allrec.tokenizer import AbstractTokenizer

class MMPGR_Tokenizer(AbstractTokenizer):
    def __init__(self, config: dict, dataset: AbstractDataset):
        super(MMPGR_Tokenizer, self).__init__(config, dataset)
        self.config = config
        self.dataset = dataset
        self.n_codebook_bits = self._get_codebook_bits(
            self.config["codebook_size"])
        self.index_factory = self._build_index_factory(config["n_codebook"])
        self.logger = getLogger()
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        self.item2tokens = self._init_tokenizer(dataset)
        self.eos_token = self.n_digit*self.codebook_size+1
        self.ignored_label = -100
    @property
    def n_digit(self):
        return self.config['n_codebook']
    @property
    def codebook_size(self):
        return self.config['codebook_size']
    @property
    def max_history_sequence_len(self):
        return self.config["max_item_seq_len"]
    @property
    def vocab_size(self) -> int:
        return self.eos_token+1
    def _build_index_factory(self, n_digit: int) -> str:
        return f"OPQ{n_digit},IVF1,PQ{n_digit}x{self.n_codebook_bits}"
    def _get_codebook_bits(self, n_codebook):
        import math
        """
        根据每个子码本大小 (codebook_size) 计算需要多少 bit 来表示一个索引.
        参数:
            n_codebook (int): 实际传入的 codebook_size, 即每个子码本的槽位数, 比如 256.
        返回:
            int: 表示一个索引所需的二进制位数 (bit 数).
        逻辑:
        - 如果一个子码本有 codebook_size 个离散槽位,
        那么就需要 log2(codebook_size) 个 bit 才能唯一表示每个槽位.
            例如:
                codebook_size = 256 -> log2(256) = 8 bit
                codebook_size = 16  -> log2(16)  = 4 bit
        - 这里要求 codebook_size 必须是 2 的幂,否则无法保证整数位数.
        """
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)
    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        items_for_training = set()
        for item_seq in dataset.split_data["train"]["item_seq"]:
            for item in item_seq:
                items_for_training.add(item)
        self.log(
            f"Items for training:{len(items_for_training)} of {dataset.n_items-1}")
        mask = np.zeros(dataset.n_items-1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask
    def _generate_semantic_id_opq(self, sent_embs, sem_ids_path, train_mask, n_digit=None, cache_label="semantic"):
        import faiss
        n_digit = int(n_digit or self.n_digit)
        train_mask = np.asarray(train_mask, dtype=bool)
        if train_mask.shape[0] != sent_embs.shape[0]:
            raise ValueError(
                f"{cache_label} train_mask shape mismatch: "
                f"mask={train_mask.shape}, embeddings={sent_embs.shape}"
            )
        if self.config['opq_use_gpu']:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = n_digit >= 56
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index_factory = self._build_index_factory(n_digit)
        index = faiss.index_factory(
            sent_embs.shape[1],
            index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log(f'[TOKENIZER] Training {cache_label} OPQ index: {index_factory}')
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(
                res, self.config['opq_gpu_id'], index, co)
        train_embs = np.ascontiguousarray(
            sent_embs[train_mask], dtype=np.float32)
        index.train(train_embs)
        sent_embs = np.ascontiguousarray(sent_embs, dtype=np.float32)
        index.add(sent_embs)
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)
        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(
            invlists.get_codes(0), ls * invlists.code_size
        )
        pq_codes = pq_codes.reshape(-1, invlists.code_size)
        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for i in range(n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)
        pq_codes = np.array(faiss_sem_ids)
        item2sem_ids = {}
        for i in range(pq_codes.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(pq_codes[i].tolist())
        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f, indent=4, ensure_ascii=False)
        return item2sem_ids
    def _compute_similarity_stats_for_embs(self, embs: np.ndarray, name: str) -> dict:
        embs = np.ascontiguousarray(embs, dtype=np.float32)
        n_items = embs.shape[0]
        if n_items <= 1:
            raise ValueError(f"{name} similarity stats require at least 2 items.")
        block_size = max(1, int(self.config.get("similarity_stats_block_size", 2048)))
        min_val = np.inf
        max_val = -np.inf
        for start in tqdm(range(0, n_items, block_size), desc=f"Computing {name} similarity stats"):
            end = min(start + block_size, n_items)
            similarity_block = np.matmul(embs[start:end], embs.T).astype(np.float32, copy=False)
            row_idx = np.arange(end - start)
            col_idx = np.arange(start, end)
            similarity_block[row_idx, col_idx] = np.nan
            block_min = np.nanmin(similarity_block)
            block_max = np.nanmax(similarity_block)
            min_val = min(min_val, float(block_min))
            max_val = max(max_val, float(block_max))
        self.log(
            f"[Tokenizer] {name} similarity stats (excluding diagonal) -> "
            f"max: {max_val:.4f}, min: {min_val:.4f}"
        )
        return {
            "min": float(min_val),
            "max": float(max_val),
        }
    def _embedding_fingerprint(self, embs: np.ndarray) -> str:
        import hashlib
        embs = np.ascontiguousarray(embs, dtype=np.float32)
        return hashlib.blake2b(embs.view(np.uint8), digest_size=16).hexdigest()
    def _prepare_realtime_similarity_targets(self, text_embs: np.ndarray, img_embs: np.ndarray, emb_dir: str):
        self.text_embs = np.ascontiguousarray(text_embs, dtype=np.float32)
        self.image_embs = np.ascontiguousarray(img_embs, dtype=np.float32)
        text_fingerprint = self._embedding_fingerprint(self.text_embs)
        image_fingerprint = self._embedding_fingerprint(self.image_embs)
        stats_file = os.path.join(
            emb_dir,
            f"{self.config['text_feature_extractor'].replace('/', '-')}_"
            f"{self.config['image_feature_extractor'].replace('/', '-')}_similarity_stats.json"
        )
        stats = None
        required_stats_keys = {"text", "image"}
        required_value_keys = {"min", "max"}
        if os.path.exists(stats_file):
            try:
                with open(stats_file, "r") as f:
                    stats = json.load(f)
                stats_valid = (
                    isinstance(stats, dict)
                    and required_stats_keys.issubset(stats.keys())
                    and all(
                        isinstance(stats[name], dict) and required_value_keys.issubset(stats[name].keys())
                        for name in required_stats_keys
                    )
                    and stats.get("text_shape") == list(self.text_embs.shape)
                    and stats.get("image_shape") == list(self.image_embs.shape)
                    and stats.get("text_fingerprint") == text_fingerprint
                    and stats.get("image_fingerprint") == image_fingerprint
                )
                if stats_valid:
                    self.log(f"[Tokenizer] Loaded cached similarity stats from {stats_file}")
                else:
                    self.log(f"[Tokenizer] Invalid similarity stats cache at {stats_file}, recomputing.", level="warning")
                    stats = None
            except Exception as e:
                self.log(f"[Tokenizer] Failed to load similarity stats from {stats_file}: {e}", level="warning")
                stats = None
        if stats is None:
            stats = {
                "text": self._compute_similarity_stats_for_embs(self.text_embs, "Text"),
                "image": self._compute_similarity_stats_for_embs(self.image_embs, "Image"),
                "text_shape": list(self.text_embs.shape),
                "image_shape": list(self.image_embs.shape),
                "text_fingerprint": text_fingerprint,
                "image_fingerprint": image_fingerprint,
            }
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)
            self.log(f"[Tokenizer] Saved similarity stats to {stats_file}")
        self.text_similarity_min = float(stats["text"]["min"])
        self.text_similarity_max = float(stats["text"]["max"])
        self.image_similarity_min = float(stats["image"]["min"])
        self.image_similarity_max = float(stats["image"]["max"])
        self.log(f"[Tokenizer] Cached text embeddings shape: {self.text_embs.shape}")
        self.log(f"[Tokenizer] Cached image embeddings shape: {self.image_embs.shape}")
    @staticmethod
    def _safe_name(value) -> str:
        return str(value).replace("/", "-")
    @staticmethod
    def _stable_hash(value) -> str:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    @staticmethod
    def _l2_normalize(embs: np.ndarray) -> np.ndarray:
        embs = np.asarray(embs, dtype=np.float32)
        return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    def _apply_pca_for_semantic(self, embs: np.ndarray, n_components: int, name: str) -> np.ndarray:
        from sklearn.decomposition import PCA
        max_components = min(embs.shape[0], embs.shape[1])
        if n_components > max_components:
            self.log(
                f"[Tokenizer] PCA_n_components for {name} exceeds max={max_components}; "
                f"use {max_components}.",
                level="warning",
            )
            n_components = max_components
        pca = PCA(n_components=n_components, whiten=True, random_state=2026)
        embs = pca.fit_transform(embs).astype(np.float32, copy=False)
        embs = self._l2_normalize(embs)
        self.log(f"[Tokenizer] PCA applied for {name}, shape:{embs.shape}")
        return embs
    def _semantic_cache_paths(self, dataset: AbstractDataset) -> dict:
        cache_dir = os.path.join(dataset.cache_dir, "processed", "MMPGR")
        os.makedirs(cache_dir, exist_ok=True)
        text_key = self._safe_name(self.config["text_feature_extractor"])
        image_key = self._safe_name(self.config["image_feature_extractor"])
        signature = {
            "version": "separate-modal-opq-v1",
            "modality": self.config["modality"],
            "text_feature_extractor": self.config["text_feature_extractor"],
            "image_feature_extractor": self.config["image_feature_extractor"],
            "use_pca": bool(self.config.get("use_pca", False)),
            "PCA_n_components": int(self.config.get("PCA_n_components", 0)),
            "n_codebook": int(self.config["n_codebook"]),
            "codebook_size": int(self.config["codebook_size"]),
            "n_items": int(dataset.n_items),
            "id2item_hash": self._stable_hash(dataset.id_mapping.get("id2item", [])),
        }
        name = (
            f"{text_key}_{image_key}_{self.config['modality']}_"
            f"separate_opq_l{self.n_digit}_k{self.codebook_size}_"
            f"{self._stable_hash(signature)}"
        )
        return {
            "final": os.path.join(cache_dir, f"{name}_sem_ids.json"),
            "text": os.path.join(cache_dir, f"{name}_text_sem_ids.json"),
            "image": os.path.join(cache_dir, f"{name}_image_sem_ids.json"),
        }
    def _load_json_if_exists(self, path: str):
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
    def _save_json(self, path: str, value) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(value, f, indent=4, ensure_ascii=False)
    def _concat_modal_semantic_ids(self, text_item2sem_ids: dict, image_item2sem_ids: dict, sem_ids_path: str) -> dict:
        item2sem_ids = {}
        for i in range(self.dataset.n_items - 1):
            item = self.id2item[i + 1]
            if item not in text_item2sem_ids or item not in image_item2sem_ids:
                raise KeyError(f"Missing modality semantic ID for item={item}")
            item2sem_ids[item] = list(text_item2sem_ids[item]) + list(image_item2sem_ids[item])
        self._save_json(sem_ids_path, item2sem_ids)
        self.log(f"[Tokenizer] Saved concatenated semantic IDs to {sem_ids_path}")
        return item2sem_ids
    def _init_tokenizer(self, dataset: AbstractDataset):
        emb_dir = os.path.join(dataset.cache_dir, "processed")
        cache_paths = self._semantic_cache_paths(dataset)
        sem_ids_path = cache_paths["final"]
        img_embs = self._encode_img_embs(dataset, emb_dir)
        text_embs = self._encode_text_embs(dataset, emb_dir)
        img_embs = self._l2_normalize(img_embs)
        text_embs = self._l2_normalize(text_embs)
        self._prepare_realtime_similarity_targets(text_embs, img_embs, emb_dir)
        item2sem_ids = self._load_json_if_exists(sem_ids_path)
        if item2sem_ids is not None:
            self.log(f"[Tokenizer] Loaded cached semantic IDs from {sem_ids_path}")
            return self._sem_ids_to_tokens(item2sem_ids)
        training_item_mask = self._get_items_for_training(dataset)
        modality = self.config["modality"]
        use_pca = bool(self.config.get("use_pca", False))
        n_components = int(self.config.get("PCA_n_components", 512))
        if modality == "multimodal":
            if self.n_digit % 2 != 0:
                raise ValueError("MMPGR multimodal semantic ID requires even n_codebook.")
            branch_digit = self.n_digit // 2
            branch_components = max(1, n_components // 2)
            text_sem_embs = (
                self._apply_pca_for_semantic(text_embs, branch_components, "text semantic ID")
                if use_pca
                else text_embs
            )
            image_sem_embs = (
                self._apply_pca_for_semantic(img_embs, branch_components, "image semantic ID")
                if use_pca
                else img_embs
            )
            text_item2sem_ids = self._load_json_if_exists(cache_paths["text"])
            if text_item2sem_ids is None:
                text_item2sem_ids = self._generate_semantic_id_opq(
                    text_sem_embs,
                    cache_paths["text"],
                    training_item_mask,
                    n_digit=branch_digit,
                    cache_label="text",
                )
            else:
                self.log(f"[Tokenizer] Loaded cached text semantic IDs from {cache_paths['text']}")
            image_item2sem_ids = self._load_json_if_exists(cache_paths["image"])
            if image_item2sem_ids is None:
                image_item2sem_ids = self._generate_semantic_id_opq(
                    image_sem_embs,
                    cache_paths["image"],
                    training_item_mask,
                    n_digit=branch_digit,
                    cache_label="image",
                )
            else:
                self.log(f"[Tokenizer] Loaded cached image semantic IDs from {cache_paths['image']}")
            item2sem_ids = self._concat_modal_semantic_ids(
                text_item2sem_ids,
                image_item2sem_ids,
                sem_ids_path,
            )
            return self._sem_ids_to_tokens(item2sem_ids)
        if modality == "text":
            all_embs = self._apply_pca_for_semantic(text_embs, n_components, "text semantic ID") if use_pca else text_embs
        elif modality == "image":
            all_embs = self._apply_pca_for_semantic(img_embs, n_components, "image semantic ID") if use_pca else img_embs
        else:
            raise ValueError(f"Invalid modality: {modality}")
        self.log(f"[Tokenizer] Semantic ID embeddings shape:{all_embs.shape}")
        item2sem_ids = self._generate_semantic_id_opq(
            all_embs,
            sem_ids_path,
            training_item_mask,
            n_digit=self.n_digit,
            cache_label=modality,
        )
        self.log(f"[Tokenizer] Saved semantic IDs to {sem_ids_path}")
        return self._sem_ids_to_tokens(item2sem_ids)
    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                tokens[digit] += self.codebook_size*digit+1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids
