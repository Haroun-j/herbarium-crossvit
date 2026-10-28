"""Plot the training histories saved in results/history_*.json.

    python -m src.plot_histories --results-dir results

Writes results/figures/learning_curves.png.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# one colour per configuration, always the same one
COLORS = {
    "A": "#E69F00", "B": "#56B4E9", "C1": "#009E73",
    "C2": "#D55E00", "O2": "#0072B2", "O2_iou": "#CC79A7",
}
PANELS = [
    ("train_loss", "train loss"),
    ("val_loss", "validation loss"),
    ("val_acc", "validation accuracy"),
    ("val_f1", "validation F1 (weighted)"),
]


def load_histories(results_dir):
    histories = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "history_*.json"))):
        tag = os.path.basename(path)[len("history_"):-len(".json")]
        with open(path) as f:
            histories[tag] = json.load(f)
    return histories


def plot(histories, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.flat, PANELS):
        for tag, hist in histories.items():
            values = hist.get(key)
            if not values:
                continue
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, marker="o", markersize=4, linewidth=1.5,
                    color=COLORS.get(tag), label=tag.replace("_iou", " + IoU loss"))
            ax.set_xticks(list(epochs))
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    histories = load_histories(args.results_dir)
    if not histories:
        raise SystemExit(f"no history_*.json in {args.results_dir}")
    out = os.path.join(args.results_dir, "figures", "learning_curves.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot(histories, out)
    print(f"{len(histories)} histories plotted to {out}")


if __name__ == "__main__":
    main()
