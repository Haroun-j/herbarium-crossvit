"""Metadata loading, stratified split and the two-image dataset."""
import random
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_metadata(client, verbose=True):
    """Fetch the metadata of every specimen. Returns a list of {"id", "label"}."""
    ids = client.list_ids()
    records, failed = [], []
    for img_id in tqdm(ids, desc="metadata", disable=not verbose):
        try:
            meta = client.get_meta(img_id)
            records.append({"id": meta["id"], "label": int(meta["meta"]["epines"])})
        except Exception:
            failed.append(img_id)
    if verbose:
        counts = Counter(r["label"] for r in records)
        print(f"{len(records)} specimens loaded, {len(failed)} failed")
        print(f"class 0 (no spines): {counts[0]}, class 1 (spines): {counts[1]}")
    return records


def stratified_split(records, ratios=(0.7, 0.15, 0.15), seed=42):
    """Shuffle each class separately and cut it with the given ratios.

    The same seed always gives the same split. This matters because the test
    split has to stay the same between src.train and src.evaluate.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    rng = random.Random(seed)
    by_class = {}
    for r in records:
        by_class.setdefault(r["label"], []).append(r)

    train, val, test = [], [], []
    for label in sorted(by_class):
        items = list(by_class[label])
        rng.shuffle(items)
        n_train = int(len(items) * ratios[0])
        n_val = int(len(items) * ratios[1])
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def class_weights(records):
    """Inverse-frequency weights, in case a weighted cross-entropy is wanted."""
    counts = Counter(r["label"] for r in records)
    total = len(records)
    return torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float32)


def eval_transform(image_size=224):
    """Resize + ImageNet normalisation. Used for training too: no augmentation
    was used for the reported runs, because a flip or rotation would have to be
    applied identically to the full sheet, the segmented image and the mask."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DualImageDataset(Dataset):
    """One item = (full sheet, segmented image, label, plant mask).

    The mask is the grey level of the segmented image scaled to [0, 1]; the
    background of that image is black, so anything above 0.5 is plant. It is
    resized with nearest neighbour so that its patch grid stays aligned with
    the patch grid of the model input.
    """

    def __init__(self, records, client, image_size=224, transform=None, preload=True, cache=None):
        self.records = records
        self.client = client
        self.image_size = image_size
        self.transform = transform or eval_transform(image_size)
        self.cache = cache if cache is not None else {}
        if preload:
            self.preload()

    def preload(self):
        for kind in ("classic", "seg"):
            missing = [r["id"] for r in self.records if f"{kind}_{r['id']}" not in self.cache]
            if not missing:
                continue
            for img_id, img in self.client.fetch_many(missing, kind=kind).items():
                self.cache[f"{kind}_{img_id}"] = img
        print(f"{len(self.cache)} images in memory")

    def _image(self, img_id, kind):
        key = f"{kind}_{img_id}"
        if key not in self.cache:
            self.cache[key] = self.client.fetch_image(img_id, kind=kind)
        return self.cache[key]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        full = self._image(rec["id"], "classic")
        seg = self._image(rec["id"], "seg")

        size = (self.image_size, self.image_size)
        seg_small = seg if seg.size == size else seg.resize(size, resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(seg_small.convert("L"), dtype=np.float32) / 255.0)

        return self.transform(full), self.transform(seg), rec["label"], mask
