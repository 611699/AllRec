from logging import getLogger
import os
import json
from typing import Any, Literal
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from allrec.dataset import AbstractDataset

class AbstractTokenizer:
    def __init__(self, config: dict, dataset: AbstractDataset):
        self.config = config
        self.dataset = dataset
        self.logger = getLogger()
        self.ignored_label = -100
        self.collate_fn = {'train': None, 'val': None, 'test': None}
        self.text_embs = None
        self.image_embs = None
    def _init_tokenizer(self):
        raise NotImplementedError('Tokenizer initialization not implemented.')
    @property
    def max_history_sequence_len(self):
        raise NotImplementedError('Maximum history sequence length not implemented.')
    def log(self, message, level='info'):
        from allrec.utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)
    def _encode_img_embs(self, dataset: AbstractDataset, emb_dir: str) -> np.ndarray:
        from transformers import SiglipModel, AutoProcessor
        from PIL import Image
        os.makedirs(emb_dir, exist_ok=True)
        img_emb_file = os.path.join(emb_dir, f"{self.config['image_feature_extractor'].replace('/', '-')}_image_emb.json")
        img_embs = None
        if os.path.exists(img_emb_file):
            with open(img_emb_file, "r") as f:
                img_embs = np.array(json.load(f), dtype=np.float32)
                self.log(
                    f"[Tokenizer] Loaded cached image embeddings {img_embs.shape} from {img_emb_file}")
            return img_embs
        else:
            images_dir = os.path.join(dataset.cache_dir, "raw", "images")
            local_ready = False
            try:
                expected = max(0, int(dataset.n_items) - 1)
                if expected > 0 and os.path.isdir(images_dir):
                    existing = [n for n in os.listdir(images_dir) if n.lower().endswith(".jpg")]
                    local_ready = len(existing) >= int(0.8 * expected)
                    if local_ready:
                        self.log(f"[Tokenizer] Local images ready ({len(existing)}/{expected} jpg) at {images_dir}, skip downloading.")
            except Exception as e:
                self.log(f"[Tokenizer] Local image readiness check failed: {e}", level="warning")
            if not local_ready:
                has_http_urls = False
                try:
                    if getattr(dataset, "item2imgUrl", None):
                        for _, url in list(dataset.item2imgUrl.items())[:1000]:
                            if isinstance(url, str) and (url.startswith("http://") or url.startswith("https://")):
                                has_http_urls = True
                                break
                except Exception:
                    has_http_urls = False
                if has_http_urls:
                    self.download_images_from_dataset(dataset)
                else:
                    self.log("[Tokenizer] No http(s) image URLs provided; will encode from local files if present, otherwise use random vectors.", level="warning")
            self.log(
                f"[Tokenizer] Extracting image embeddings with {self.config['image_feature_extractor']} ...")
            image_feature_extractor = self.config['image_feature_extractor']
            processor = AutoProcessor.from_pretrained(image_feature_extractor)
            siglip_model = SiglipModel.from_pretrained(
                image_feature_extractor).to(self.config['device'])
            siglip_model.eval()
            emb_dim = siglip_model.config.vision_config.hidden_size
            img_embs = []
            for i in tqdm(range(1, dataset.n_items), desc="Encoding images"):
                img_path = os.path.join(
                    dataset.cache_dir, "raw", "images", f"{i}.jpg")
                if not os.path.exists(img_path):
                    img_embs.append(np.random.randn(
                        emb_dim).astype(np.float32))
                    self.log(
                        f"[Tokenizer] Image not found: {img_path}, using random vector.")
                    continue
                try:
                    image = Image.open(img_path).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(self.config["device"])
                    with torch.no_grad():
                        features = siglip_model.get_image_features(**inputs)
                        if hasattr(features, 'pooler_output'):
                            img_emb = features.pooler_output
                        else:
                            img_emb = features.last_hidden_state
                        img_embs.append(img_emb.cpu().numpy().squeeze())
                except Exception as e:
                    self.log(
                        f"[Tokenizer] Failed to process image {img_path}:{e}")
                    img_embs.append(np.random.randn(
                        emb_dim).astype(np.float32))
                    self.log(
                        f"[Tokenizer] Failed to process image {img_path}, using random vector.")
            img_embs = np.array(img_embs, dtype=np.float32)
            del siglip_model
            del processor
            with open(img_emb_file, "w") as f:
                json.dump(img_embs.tolist(), f, indent=4, ensure_ascii=False)
            self.log(
                f"[Tokenizer] Saved image embeddings to {img_emb_file}")
            self.log(
                f"[Tokenizer] Saved Image embeddings shape:{img_embs.shape}")
            return img_embs
    def _encode_text_embs(self, dataset: AbstractDataset, emb_dir: str) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        self.log(f"[Tokenizer] begin Encoding text embedding")
        os.makedirs(emb_dir, exist_ok=True)
        text_emb_file = os.path.join(emb_dir, f"{self.config['text_feature_extractor'].replace('/', '-')}_text_emb.json")
        text_embs = None
        if os.path.exists(text_emb_file):
            with open(text_emb_file, "r") as f:
                text_embs = np.array(json.load(f), dtype=np.float32)
                self.log(f"[Tokenizer] Loaded cached text embeddings {text_embs.shape} from {text_emb_file}")
                return text_embs
        else:
            self.log(
                f"[Tokenizer] Extracting text embeddings with {self.config['text_feature_extractor']} ...")
            missing_meta_count = 0
            meta_sentences = []
            for i in range(1, dataset.n_items):
                asin = dataset.id_mapping['id2item'][i]
                meta_content = dataset.item2meta.get(asin) if asin else None
                if meta_content:
                    meta_sentences.append(meta_content)
                else:
                    meta_sentences.append("")
                    missing_meta_count += 1
            self.log(f"[Tokenizer] Text metadata stats: total items {dataset.n_items - 1}, "
                    f"missing text metadata {missing_meta_count}, "
                    f"fallback ratio {missing_meta_count / (dataset.n_items - 1):.2%}")
            sent_emb_model = SentenceTransformer(self.config['text_feature_extractor']).to(self.config["device"])
            text_embs = sent_emb_model.encode(
                meta_sentences,
                convert_to_numpy=True,
                batch_size=64,
                show_progress_bar=True,
                device=self.config["device"]
            )
            del sent_emb_model
            with open(text_emb_file, "w") as f:
                json.dump(text_embs.tolist(), f, indent=4, ensure_ascii=False)
            self.log(
                f"[Tokenizer] Saved text embeddings to {text_emb_file}")
            self.log(
                f"[Tokenizer] Saved text embeddings shape:{text_embs.shape}")
            return text_embs
    def download_images_from_dataset(self, dataset: AbstractDataset):
        import time
        item2imgUrl = dataset.item2imgUrl
        item2id = dataset.id_mapping['item2id']
        save_dir = os.path.join(dataset.cache_dir, "raw", "images")
        timeout = 50
        num_threads = 200
        max_retries = 3
        os.makedirs(save_dir, exist_ok=True)
        failed = []
        def download_one(asin, url) :
            item_id = item2id.get(asin, None)
            if item_id is None:
                return asin, "No item_id mapping"
            if url is None:
                return asin, "No URL"
            ext = os.path.splitext(url)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".gif"]:
                ext = ".jpg"
            filename = os.path.join(save_dir, f"{item_id}{ext}")
            if os.path.exists(filename):
                return None
            last_error = ""
            for attempt in range(max_retries):
                try:
                    resp = requests.get(url, timeout=timeout)
                    if resp.status_code == 200:
                        if len(resp.content) > 0:
                            with open(filename, "wb") as f:
                                f.write(resp.content)
                            return None
                    elif resp.status_code == 404:
                        last_error = "HTTP 404 (Not Found)"
                        self.log(f"[Tokenizer] Image not found: {url}")
                        return asin, last_error
                    else:
                        last_error = f"HTTP {resp.status_code}"
                except Exception as e:
                    last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(attempt + 1)
                else:
                    self.log(f"[Tokenizer] Failed after {max_retries} attempts,url:{url},Reason:{last_error}")
                    return asin, last_error
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(download_one, asin, url): asin
                for asin, url in item2imgUrl.items()
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
                result = future.result()
                if result:
                    failed.append(result)
        if failed:
            fail_file = os.path.join(save_dir, "failed.txt")
            with open(fail_file, "w", encoding="utf-8") as f:
                for asin, reason in failed:
                    f.write(f"{asin}\t{reason}\n")
            self.log(f"[Tokenizer] Failed to download {len(failed)} images; details saved to {fail_file}")
        else:
            self.log("[Tokenizer] All images were downloaded successfully or already exist.")
        all_files = os.listdir(save_dir)
        total_images = len(all_files)
        jpg_images = len([f for f in all_files if f.lower().endswith(".jpg")])
        expected = len(item2imgUrl)
        from collections import Counter
        ext_counter = Counter(os.path.splitext(
            f)[1].lower() for f in all_files)
        self.log("\n===== Download statistics =====")
        self.log(f"[Tokenizer] Expected item count: {expected}")
        self.log(f"[Tokenizer] Image file count: {total_images}")
        self.log(f"[Tokenizer] JPG file count: {jpg_images}")
        self.log("\n[Tokenizer] File extension distribution:")
        for ext, count in ext_counter.items():
            self.log(f"   {ext or '[no extension]'}: {count}")
        if total_images == expected:
            self.log("[Tokenizer] Image count matches item count.")
        else:
            self.log("[Tokenizer] Image count does not match item count; downloads may have failed.")
        if jpg_images == expected:
            self.log("[Tokenizer] JPG file count matches item count.")
        else:
            self.log("[Tokenizer] JPG file count does not match item count; files may be missing or non-JPG.")
    def _compute_similarity_matrix(self, text_embs: np.ndarray, image_embs: np.ndarray, dir: str):
        import numpy as np
        import os
        text_similarity_matrix_file = os.path.join(dir, f"{self.config['text_feature_extractor'].replace('/', '-')}_text_similarity_matrix.npy")
        image_similarity_matrix_file = os.path.join(dir, f"{self.config['image_feature_extractor'].replace('/', '-')}_image_similarity_matrix.npy")
        if os.path.exists(text_similarity_matrix_file) and os.path.exists(image_similarity_matrix_file):
            text_similarity_matrix = np.load(text_similarity_matrix_file)
            image_similarity_matrix = np.load(image_similarity_matrix_file)
            self.log(f"[Tokenizer] Loaded cached similarity matrices from {text_similarity_matrix_file} and {image_similarity_matrix_file}")
            return text_similarity_matrix, image_similarity_matrix
        self.log("[Tokenizer] Computing similarity matrices ...")
        text_similarity_matrix_ori :np.ndarray= np.matmul(text_embs, text_embs.T)
        image_similarity_matrix_ori :np.ndarray= np.matmul(image_embs, image_embs.T)
        text_similarity_matrix :np.ndarray= text_similarity_matrix_ori.astype(np.float32)
        image_similarity_matrix :np.ndarray= image_similarity_matrix_ori.astype(np.float32)
        def compute_stats(matrix, name: str):
            matrix_no_diag = matrix.copy()
            np.fill_diagonal(matrix_no_diag, np.nan)
            max_val = np.nanmax(matrix_no_diag)
            min_val = np.nanmin(matrix_no_diag)
            mean_val = np.nanmean(matrix_no_diag)
            std_val = np.nanstd(matrix_no_diag)
            self.log(f"[Tokenizer] {name} (excluding diagonal) -> "
                    f"max: {max_val:.4f}, min: {min_val:.4f}, mean: {mean_val:.4f}, std: {std_val:.4f}")
            return min_val, max_val
        text_min, text_max = compute_stats(text_similarity_matrix, "Text similarity matrix")
        image_min, image_max = compute_stats(image_similarity_matrix, "Image similarity matrix")
        text_similarity_matrix =2*(text_similarity_matrix - text_min) / (text_max - text_min + 1e-12)-1
        image_similarity_matrix =2*(image_similarity_matrix - image_min) / (image_max - image_min + 1e-12)-1
        np.save(text_similarity_matrix_file, text_similarity_matrix)
        np.save(image_similarity_matrix_file, image_similarity_matrix)
        self.log(f"[Tokenizer] Saved normalized similarity matrices to:")
        self.log(f"  - {text_similarity_matrix_file}")
        self.log(f"  - {image_similarity_matrix_file}")
        self.log(f"[Tokenizer] Normalized text matrix -> range: [{text_similarity_matrix.min():.4f}, {text_similarity_matrix.max():.4f}]")
        self.log(f"[Tokenizer] Normalized image matrix -> range: [{image_similarity_matrix.min():.4f}, {image_similarity_matrix.max():.4f}]")
        return text_similarity_matrix, image_similarity_matrix
    def tokenize_datasets(self, datasets: dict) -> dict:
        from datasets import load_from_disk
        tokenized_datasets = {}
        cache_name = f"{self.config['tokenizer_datasets_mode']}_len{self.max_history_sequence_len}"
        save_dir=os.path.join(self.dataset.cache_dir, "processed","tokenized_datasets",cache_name)
        os.makedirs(save_dir,exist_ok=True)
        for split in datasets:
            save_path=os.path.join(save_dir,split)
            if os.path.exists(save_path):
                tokenized_datasets[split] = load_from_disk(save_path)
                self.log(f"[Tokenizer] Loaded cached {cache_name} tokenized datasets from {save_path}")
                continue
            tokenize_function = self.compression_tokenize_function if self.config['tokenizer_datasets_mode'] == "compression" else self.slide_window_tokenize_function
            tokenized_datasets[split] = datasets[split].map(
                lambda t: tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=None,
                desc=f'Tokenizing {split} set'
            )
            tokenized_datasets[split].save_to_disk(save_path)
            self.log(f"[Tokenizer] Saved {cache_name} tokenized datasets to {save_path}")
        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')
        return tokenized_datasets
    def compression_tokenize_function(self, example: dict, split: str) -> dict:
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]
        if split == 'train':
            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)
            head_seq = item_seq[:min(len(item_seq), max_item_seq_len + 1)]
            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                head_seq)
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
                [input_ids], [attention_mask], [labels], [seq_lens]
            for i in range(1, n_return_examples):
                cur_seq = item_seq[i:i+max_item_seq_len+1]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                    cur_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)
            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'seq_lens': all_seq_lens,
            }
        else:
            tail_seq = item_seq[-(max_item_seq_len+1):]
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                tail_seq, pad_labels=False
            )
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels[-1:]],
                'seq_lens': [seq_lens],
            }
    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        pad_lens = self.max_history_sequence_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        labels = [self.item2id[item] for item in item_seq[1:]]
        labels.extend([self.ignored_label] * pad_lens)
        return input_ids, attention_mask, labels, seq_lens
    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        if seq_lens == 0:
            self.log(f"[Tokenizer] Warning: seq_lens is 0, item_seq: {item_seq}")
            self.log(f"[Tokenizer] input_ids: {input_ids}")
            self.log(f"[Tokenizer] attention_mask: {attention_mask}")
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2id[item_seq[-1]]
        pad_lens = self.max_history_sequence_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)
        return input_ids, attention_mask, labels, seq_lens
    def slide_window_tokenize_function(self, example: dict, split: str) -> dict:
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]
        if split == "train":
            all_input_ids = []
            all_attention_mask = []
            all_labels = []
            all_seq_lens = []
            for i in range(1, len(item_seq)):
                start_idx = max(0, i - max_item_seq_len)
                cur_seq = item_seq[start_idx:i+1]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                    cur_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)
            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'seq_lens': all_seq_lens,
            }
        else:
            tail_seq = item_seq[-(max_item_seq_len+1):]
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                tail_seq, pad_labels=False)
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels[-1:]],
                'seq_lens': [seq_lens],
            }
