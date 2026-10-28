"""Evaluate the saved checkpoints on the held-out test split.

    python -m src.evaluate --results-dir results

Prints a classification report per model, writes results/test_metrics.json
and one confusion matrix per model in results/figures/.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .api_client import HerbariumClient
from .config import ModelConfig, TrainConfig
from .data import DualImageDataset, load_metadata, stratified_split
from .interpretability import attention_rollout, batch_attention_iou
from .train import build_model

CLASS_NAMES = ["no spines", "spines"]


def find_checkpoints(results_dir):
    """{tag: path} for every results/best_model_<tag>.pth, e.g. 'O2', 'O2_iou'."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "best_model_*.pth"))):
        tag = os.path.basename(path)[len("best_model_"):-len(".pth")]
        out[tag] = path
    return out


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels_all, ious = [], [], []
    for full, seg, labels, masks in loader:
        logits, attentions = model(full.to(device), seg.to(device), return_attention=True)
        ious += batch_attention_iou(attention_rollout(attentions), masks)
        preds += logits.argmax(1).cpu().tolist()
        labels_all += labels.tolist()
    return np.array(labels_all), np.array(preds), np.array(ious)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=42, help="must match the seed used for training")
    args = parser.parse_args()

    mcfg, tcfg = ModelConfig(), TrainConfig(results_dir=args.results_dir, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = find_checkpoints(tcfg.results_dir)
    if not checkpoints:
        raise SystemExit(f"no best_model_*.pth in {tcfg.results_dir}, run src.train first")

    client = HerbariumClient(cache_dir=tcfg.cache_dir)
    _, _, test_recs = stratified_split(load_metadata(client), seed=tcfg.seed)
    test_ds = DualImageDataset(test_recs, client, mcfg.image_size)
    loader = DataLoader(test_ds, batch_size=tcfg.batch_size, shuffle=False)
    print(f"{len(test_ds)} test images")

    figures_dir = os.path.join(tcfg.results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    metrics = {}
    for tag, path in checkpoints.items():
        model = build_model(tag.replace("_iou", ""), mcfg).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        y, y_hat, ious = predict(model, loader, device)

        cm = confusion_matrix(y, y_hat)
        metrics[tag] = {
            "accuracy": float(accuracy_score(y, y_hat)),
            "f1_weighted": float(f1_score(y, y_hat, average="weighted")),
            "attention_iou_mean": float(ious.mean()),
            "attention_iou_std": float(ious.std()),
            "confusion_matrix": cm.tolist(),
        }
        print(f"\n=== {tag} ===")
        print(classification_report(y, y_hat, target_names=CLASS_NAMES))
        print(f"attention IoU: {ious.mean():.3f} +/- {ious.std():.3f}")
        plot_confusion(cm, f"{tag} (test)", os.path.join(figures_dir, f"test_confusion_{tag}.png"))

    with open(os.path.join(tcfg.results_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nmodel      acc     F1      IoU")
    for tag, m in sorted(metrics.items(), key=lambda kv: -kv[1]["f1_weighted"]):
        print(f"{tag:<8} {m['accuracy']:.4f}  {m['f1_weighted']:.4f}  {m['attention_iou_mean']:.3f}")


if __name__ == "__main__":
    main()
