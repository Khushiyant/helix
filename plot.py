import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "raw": "#2196F3",
    "helix-raw": "#F44336",
    "spectrogram": "#4CAF50",
    "helix-spectrogram": "#FF9800",
}

LABELS = {
    "raw": "Pure Mamba (Raw)",
    "helix-raw": "HELIX (Raw)",
    "spectrogram": "Pure Mamba (Spec)",
    "helix-spectrogram": "HELIX (Spec)",
}


def load_results(results_dir, modes):
    data = {}
    for mode in modes:
        path = os.path.join(results_dir, f"{mode}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                data[mode] = json.load(f)
        else:
            print(f"Warning: {path} not found, skipping")
    return data


def plot_training_curves(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for mode, result in data.items():
        color = COLORS.get(mode, "#999999")
        label = LABELS.get(mode, mode)
        histories = result["histories"]

        all_losses = []
        all_accs = []
        for fold_key, history in histories.items():
            epochs = [h["epoch"] for h in history]
            all_losses.append([h["test_loss"] for h in history])
            all_accs.append([h["test_acc"] for h in history])

        mean_loss = np.mean(all_losses, axis=0)
        std_loss = np.std(all_losses, axis=0)
        mean_acc = np.mean(all_accs, axis=0)
        std_acc = np.std(all_accs, axis=0)

        axes[0].plot(epochs, mean_loss, color=color, label=label, linewidth=2)
        axes[0].fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.15)

        axes[1].plot(epochs, mean_acc, color=color, label=label, linewidth=2)
        axes[1].fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.15)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test Loss")
    axes[0].set_title("Test Loss (mean ± std across folds)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].set_title("Test Accuracy (mean ± std across folds)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_fold_comparison(data, out_dir):
    modes = list(data.keys())
    n_modes = len(modes)
    n_folds = max(len(data[m]["fold_accuracies"]) for m in modes)
    x = np.arange(n_folds)
    width = 0.8 / n_modes

    fig, ax = plt.subplots(figsize=(max(8, n_folds * 1.5), 5))
    for i, mode in enumerate(modes):
        accs = data[mode]["fold_accuracies"]
        color = COLORS.get(mode, "#999999")
        label = LABELS.get(mode, mode)
        offset = (i - n_modes / 2 + 0.5) * width
        bars = ax.bar(x[:len(accs)] + offset, accs, width, label=label, color=color, alpha=0.85)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{acc:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_title("Per-Fold Best Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fold_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_summary(data, out_dir):
    modes = list(data.keys())
    means = [data[m]["mean_accuracy"] for m in modes]
    stds = [data[m]["std_accuracy"] for m in modes]
    colors = [COLORS.get(m, "#999999") for m in modes]
    labels = [LABELS.get(m, m) for m in modes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.85,
                  capsize=8, edgecolor="black", linewidth=0.5)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.5,
                f"{mean:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Mean Accuracy (%)")
    n_folds = max(len(data[m]["fold_accuracies"]) for m in modes)
    ax.set_title(f"{n_folds}-Fold Cross Validation Results")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="plots")
    parser.add_argument("--modes", nargs="+",
                        default=["raw", "helix-raw"],
                        help="Which result files to plot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_results(args.results_dir, args.modes)
    if not data:
        print("No results found. Run training first.")
        return

    plot_training_curves(data, args.out_dir)
    plot_fold_comparison(data, args.out_dir)
    plot_summary(data, args.out_dir)
    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
