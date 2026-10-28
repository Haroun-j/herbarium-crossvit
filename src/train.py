"""Train one or several CrossViT configurations.

    python -m src.train --modes A B C1 C2 O2 --epochs 5
    python -m src.train --modes O2 --iou-loss

Each run saves results/best_model_<mode>.pth (best validation F1) and
results/history_<mode>.json. The test split is never touched here, it is
used by src.evaluate only. A GPU is needed in practice: on an H100, 5 epochs
of one configuration took a few minutes; on CPU it is hours.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .api_client import HerbariumClient
from .config import ModelConfig, TrainConfig
from .data import DualImageDataset, load_metadata, stratified_split
from .interpretability import attention_rollout, batch_attention_iou
from .losses import IoUWeightedLoss
from .model import DualImageCrossViT, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(mode, cfg: ModelConfig):
    return DualImageCrossViT(
        num_classes=cfg.num_classes, img_size=cfg.image_size,
        patch_size_small=cfg.patch_size_small, patch_size_large=cfg.patch_size_large,
        embed_dim_small=cfg.embed_dim_small, embed_dim_large=cfg.embed_dim_large,
        depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
        qkv_bias=cfg.qkv_bias, drop_rate=cfg.dropout, attn_drop_rate=cfg.attention_dropout,
        mode=mode,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, measure_iou=True):
    """Loss, accuracy, weighted F1 and mean attention IoU on a loader."""
    model.eval()
    total_loss, preds, labels_all, ious = 0.0, [], [], []
    for full, seg, labels, masks in loader:
        full, seg, labels = full.to(device), seg.to(device), labels.to(device)
        logits, attentions = model(full, seg, return_attention=True)
        rollout = attention_rollout(attentions)
        if isinstance(criterion, IoUWeightedLoss):
            loss, _, _ = criterion(logits, labels, rollout, masks)
        else:
            loss = criterion(logits, labels)
        if measure_iou:
            ious += batch_attention_iou(rollout, masks)
        total_loss += loss.item()
        preds += logits.argmax(1).cpu().tolist()
        labels_all += labels.cpu().tolist()
    return {
        "loss": total_loss / max(len(loader), 1),
        "acc": accuracy_score(labels_all, preds),
        "f1": f1_score(labels_all, preds, average="weighted"),
        "iou": float(np.mean(ious)) if ious else None,
    }


def train_one_config(mode, train_loader, val_loader, mcfg, tcfg, use_iou_loss=False, device="cuda"):
    tag = f"{mode}_iou" if use_iou_loss else mode
    model = build_model(mode, mcfg).to(device)
    print(f"[{tag}] {count_parameters(model) / 1e6:.1f}M parameters")

    criterion = IoUWeightedLoss(tcfg.lambda_iou) if use_iou_loss else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.epochs, eta_min=tcfg.min_lr)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "val_iou")}
    best_f1 = 0.0
    checkpoint = os.path.join(tcfg.results_dir, f"best_model_{tag}.pth")

    for epoch in range(tcfg.epochs):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        for full, seg, labels, masks in tqdm(train_loader, desc=f"{tag} epoch {epoch + 1}/{tcfg.epochs}"):
            full, seg, labels = full.to(device), seg.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_iou_loss:
                logits, attentions = model(full, seg, return_attention=True)
                loss, _, _ = criterion(logits, labels, attention_rollout(attentions), masks)
            else:
                logits = model(full, seg)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        val = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(running_loss / len(train_loader))
        history["train_acc"].append(correct / seen)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        history["val_f1"].append(val["f1"])
        history["val_iou"].append(val["iou"])
        print(f"[{tag}] epoch {epoch + 1}: train loss {history['train_loss'][-1]:.4f}, "
              f"train acc {history['train_acc'][-1]:.4f}, val loss {val['loss']:.4f}, "
              f"val acc {val['acc']:.4f}, val F1 {val['f1']:.4f}, val IoU {val['iou']:.4f}")

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save(model.state_dict(), checkpoint)
        scheduler.step()

    with open(os.path.join(tcfg.results_dir, f"history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history, best_f1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=["A", "B", "C1", "C2", "O2"],
                        choices=DualImageCrossViT.MODES)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--iou-loss", action="store_true", help="add the IoU penalty to the loss (O5)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    mcfg = ModelConfig()
    tcfg = TrainConfig(batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.lr,
                       seed=args.seed, results_dir=args.results_dir)
    set_seed(tcfg.seed)
    os.makedirs(tcfg.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    client = HerbariumClient(cache_dir=tcfg.cache_dir)
    records = load_metadata(client)
    train_recs, val_recs, test_recs = stratified_split(records, seed=tcfg.seed)
    print(f"split: {len(train_recs)} train / {len(val_recs)} val / {len(test_recs)} test (test kept for src.evaluate)")

    cache = {}
    train_ds = DualImageDataset(train_recs, client, mcfg.image_size, cache=cache)
    val_ds = DualImageDataset(val_recs, client, mcfg.image_size, cache=cache)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,
                              num_workers=tcfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False,
                            num_workers=tcfg.num_workers, pin_memory=True)

    summary = {}
    for mode in args.modes:
        _, best = train_one_config(mode, train_loader, val_loader, mcfg, tcfg, args.iou_loss, device)
        summary[mode] = best
    print("best validation F1 per configuration:")
    for mode, f1 in summary.items():
        print(f"  {mode:>3}: {f1:.4f}")


if __name__ == "__main__":
    main()
