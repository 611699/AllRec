import random
import os
import gzip
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from typing import Optional
from allrec.dataset import AbstractDataset
from allrec.utils import download_file, clean_text

class AmazonReviews2014(AbstractDataset):
    def __init__(self, config: dict):
        super(AmazonReviews2014, self).__init__(config)
        self.category = config['category']
        self._check_available_category()
        self.log(f'[Dataset] Amazon Reviews 2014 category: {self.category}')
        self.cache_dir = os.path.join(config['cache_dir'], 'AmazonReviews2014', self.category)
        self._download_and_process_raw()
    def _check_available_category(self):
        available_categories = [
            'Home_and_Kitchen',
            'Sports_and_Outdoors',
            'Toys_and_Games',
            'Beauty'
        ]
        assert self.category in available_categories, \
            f'Category "{self.category}" not available. ' \
            f'Available categories: {available_categories}'
    def _download_raw(self, path: str, type: str = 'reviews') -> str:
        url = f'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{type}_{self.category}{"_5" if type == "reviews" else ""}.json.gz'
        base_name = os.path.basename(url)
        local_filepath = os.path.join(path, base_name)
        if not os.path.exists(local_filepath):
            download_file(url, local_filepath)
        return local_filepath
    def _parse_gz(self, path: str):
        g = gzip.open(path, 'r')
        for l in g:
            l = l.replace(b'true', b'True').replace(b'false', b'False')
            yield eval(l)
    def _load_reviews(self, path: str) -> list:
        self.log('[Dataset] Loading reviews...')
        reviews = []
        total_lines = sum(1 for _ in self._parse_gz(path))
        for inter in tqdm(self._parse_gz(path), total=total_lines, desc="Loading reviews"):
            user = inter['reviewerID']
            item = inter['asin']
            time = inter['unixReviewTime']
            reviews.append((user, item, int(time)))
        return reviews
    def _get_item_seqs(self, reviews: list[tuple]) -> dict:
        item_seqs = defaultdict(list)
        for data in reviews:
            user, item, time = data
            item_seqs[user].append((item, time))
        for user, item_time in item_seqs.items():
            item_time.sort(key=lambda x: x[1])
            item_seqs[user] = [item for item, _ in item_time]
        return item_seqs
    def _remap_ids(self, item_seqs: dict) -> tuple[dict, dict]:
        self.log('[Dataset] Remapping user and item IDs...')
        for user, items in item_seqs.items():
            if user not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
                self.id_mapping['id2user'].append(user)
            items_list = []
            for item in items:
                if item not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
                    self.id_mapping['id2item'].append(item)
                items_list.append(item)
            self.all_item_seqs[user] = items_list
        return self.all_item_seqs, self.id_mapping
    def _process_reviews(
        self,
        input_path: str,
        output_path: str
    ) -> tuple[dict, dict]:
        seq_file = os.path.join(output_path, 'all_item_seqs.json')
        id_mapping_file = os.path.join(output_path, 'id_mapping.json')
        if os.path.exists(seq_file) and os.path.exists(id_mapping_file):
            self.log('[Dataset] Reviews already processed; loading cache.')
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            return all_item_seqs, id_mapping
        self.log('[Dataset] Processing reviews...')
        reviews = self._load_reviews(input_path)
        item_seqs = self._get_item_seqs(reviews)
        all_item_seqs, id_mapping = self._remap_ids(item_seqs)
        self.log('[Dataset] Saving mapping data...')
        with open(seq_file, 'w') as f:
            json.dump(all_item_seqs, f, indent=4, ensure_ascii=False)
        with open(id_mapping_file, 'w') as f:
            json.dump(id_mapping, f, indent=4, ensure_ascii=False)
        return all_item_seqs, id_mapping
    def _load_metadata(
            self,
            path: str,
            item2id: dict
        ) -> dict:
        self.log('[Dataset] Loading metadata...')
        data = {}
        item_asins = set(item2id.keys())
        total_lines = sum(1 for _ in self._parse_gz(path))
        for info in tqdm(self._parse_gz(path), total=total_lines, desc="Loading metadata"):
            if info['asin'] not in item_asins:
                continue
            data[info['asin']] = info
        return data
    def _sent_process(self, raw: str) -> str:
        sentence = ""
        if isinstance(raw, float):
            sentence += str(raw)
            sentence += '.'
        elif len(raw) > 0 and isinstance(raw[0], list):
            for v1 in raw:
                for v in v1:
                    sentence += clean_text(v)[:-1]
                    sentence += ', '
            sentence = sentence[:-2]
            sentence += '.'
        elif isinstance(raw, list):
            for v1 in raw:
                sentence += clean_text(v1)
        else:
            sentence = clean_text(raw)
        return sentence + ' '
    def _extract_meta_sentences(
        self,
        metadata: dict
    ) -> dict:
        self.log('[Dataset] Extracting meta sentences with clear keywords...')
        item2meta = {}
        total_items = len(metadata)
        for item, meta in tqdm(metadata.items(), total=total_items, desc="Extracting meta sentences", unit="item"):
            lines = []
            if "title" in meta:
                lines.append("[Title]:")
                lines.append(self._sent_process(meta["title"]).strip()+"\n")
            if "price" in meta:
                lines.append("[Price]:")
                lines.append(self._sent_process(meta["price"]).strip()+"\n")
            if "brand" in meta:
                lines.append("[Brand]:")
                lines.append(self._sent_process(meta["brand"]).strip()+"\n")
            if "feature" in meta:
                lines.append("[Feature]:")
                lines.append(self._sent_process(meta["feature"]).strip()+"\n")
            if "categories" in meta:
                lines.append("[Categories]:")
                lines.append(self._sent_process(meta["categories"]).strip()+"\n")
            meta_sentence = "  ".join(lines)
            item2meta[item] = meta_sentence
        return item2meta
    def _process_meta(
        self,
        input_path: str,
        output_path: str
    ) -> Optional[dict]:
        coretext_file = os.path.join(output_path, f'metadata.coretext.json')
        asin_img_file = os.path.join(output_path, 'metadata.asin_img.json')
        allmeta_file = os.path.join(output_path, 'metadata.allmeta.json')
        if os.path.exists(coretext_file) and os.path.exists(asin_img_file) and os.path.exists(allmeta_file):
            self.log('[Dataset] Metadata already processed; loading cache.')
            with open(coretext_file, 'r') as f:
                item2coretext = json.load(f)
            with open(asin_img_file, "r") as f:
                item2imgUrl = json.load(f)
            with open(allmeta_file, "r") as f:
                item2allmeta = json.load(f)
            return item2coretext, item2imgUrl, item2allmeta
        self.log('[Dataset] Loading metadata from raw file...')
        item2allmeta = self._load_metadata(
            path=input_path,
            item2id=self.item2id
        )
        def get_asin_img(item2allmeta, output_path):
            category2imgs = {}
            missing_imurl_asins = []
            missing_category_asins = []
            for asin, meta in item2allmeta.items():
                im_url = meta.get("imUrl", None)
                categories = meta.get("categories", [])
                if im_url:
                    if categories:
                        cat_tuple = tuple(categories[0]) if isinstance(categories[0], list) else tuple(categories)
                        category2imgs.setdefault(cat_tuple, []).append(im_url)
            filtered_meta = {}
            for asin, meta in item2allmeta.items():
                im_url = meta.get("imUrl", None)
                categories = meta.get("categories", [])
                if not im_url:
                    missing_imurl_asins.append(asin)
                    candidates = []
                    if categories:
                        cat_tuple = tuple(categories[0]) if isinstance(categories[0], list) else tuple(categories)
                        candidates = category2imgs.get(cat_tuple, [])
                    if candidates:
                        im_url = random.choice(candidates)
                    else:
                        missing_category_asins.append(asin)
                if im_url:
                    filtered_meta[asin] = {"asin": asin, "imUrl": im_url}
            item2imgUrl = {asin: meta["imUrl"] for asin, meta in filtered_meta.items()}
            save_file = os.path.join(output_path, "metadata.asin_img.json")
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(filtered_meta, f, indent=4, ensure_ascii=False)
            print(f"[Dataset] Saved {len(filtered_meta)} asin+imUrl records to {save_file}")
            print(f"[Dataset] Missing imUrl records: {len(missing_imurl_asins)}")
            print(f"[Dataset] Same-category image fills: {len(missing_imurl_asins) - len(missing_category_asins)}")
            print(f"[Dataset] Records without same-category image candidates: {len(missing_category_asins)}")
            return item2imgUrl
        item2imgUrl = get_asin_img(item2allmeta, output_path)
        item2coretext = self._extract_meta_sentences(metadata=item2allmeta)
        with open(coretext_file, 'w') as f:
            json.dump(item2coretext, f, indent=4, ensure_ascii=False)
        with open(asin_img_file, "w") as f:
            json.dump(item2imgUrl, f, indent=4, ensure_ascii=False)
        with open(allmeta_file, "w") as f:
            json.dump(item2allmeta, f, indent=4, ensure_ascii=False)
        return item2coretext, item2imgUrl, item2allmeta
    def _download_and_process_raw(self):
        raw_data_path = os.path.join(self.cache_dir, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        with self.accelerator.main_process_first():
            reviews_localpath = self._download_raw(
                path=raw_data_path,
                type='reviews'
            )
            meta_localpath = self._download_raw(
                path=raw_data_path,
                type='meta'
            )
        np.random.seed(12345)
        processed_data_path = os.path.join(self.cache_dir, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)
        self.all_item_seqs, self.id_mapping = self._process_reviews(
            input_path=reviews_localpath,
            output_path=processed_data_path
        )
        self.item2meta, self.item2imgUrl, self.item2allmeta = self._process_meta(
            input_path=meta_localpath,
            output_path=processed_data_path
        )
