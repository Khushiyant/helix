import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import RawWaveformMamba, SpectrogramMamba
from dataset import get_dataloaders, get_speechcommands_dataloaders

N_LAYERS = 6

DATASET_CONFIG = {
    "esc50": {"num_classes": 50, "num_folds": 5, "default_root": "data/ESC-50-master"},
    "speechcommands": {"num_classes": 35, "num_folds": 1, "default_root": "data"},
}


def mixup(x, y, alpha=0.3):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)

    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=True):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup(inputs, targets, alpha=0.3)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        total += targets.size(0)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_fold(fold, mode, data_root, device, dataset="esc50", epochs=100, batch_size=32, lr=3e-4):
    cfg = DATASET_CONFIG[dataset]
    num_folds = cfg["num_folds"]
    num_classes = cfg["num_classes"]

    print(f"\n{'='*60}")
    print(f"  FOLD {fold}/{num_folds} — Mode: {mode} — Dataset: {dataset}")
    print(f"{'='*60}")

    data_mode = mode.replace("helix-", "").replace("attention-", "")
    loader_fn = get_speechcommands_dataloaders if dataset == "speechcommands" else get_dataloaders
    train_loader, test_loader = loader_fn(
        root=data_root,
        test_fold=fold,
        batch_size=batch_size,
        mode=data_mode,
    )

    pure_attention = mode.startswith("attention-")
    helix = mode.startswith("helix-")
    base_mode = mode.replace("helix-", "").replace("attention-", "")

    if pure_attention:
        attention_at = tuple(range(N_LAYERS))
    elif helix:
        attention_at = (3,)
    else:
        attention_at = ()

    if base_mode == "raw":
        model = RawWaveformMamba(num_classes=num_classes, n_layers=N_LAYERS, attention_at=attention_at).to(device)
    else:
        model = SpectrogramMamba(num_classes=num_classes, n_layers=N_LAYERS, attention_at=attention_at).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_prefix = f"{dataset}_{mode}" if dataset != "esc50" else mode
            torch.save(
                model.state_dict(),
                f"checkpoints/{ckpt_prefix}_fold{fold}_best.pt"
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.4f} / {train_acc:.1f}% | "
                f"Test: {test_loss:.4f} / {test_acc:.1f}% | "
                f"Best: {best_acc:.1f}% | "
                f"{elapsed:.1f}s"
            )

    print(f"\n  Fold {fold} best accuracy: {best_acc:.1f}%")
    return best_acc, history


def run_experiment(mode, data_root, device, dataset="esc50", epochs=100, batch_size=32, lr=3e-4):
    cfg = DATASET_CONFIG[dataset]
    num_folds = cfg["num_folds"]

    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT: {mode.upper()} — Dataset: {dataset}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"{'#'*60}")

    fold_accuracies = []
    all_histories = {}

    for fold in range(1, num_folds + 1):
        best_acc, history = train_fold(
            fold=fold,
            mode=mode,
            data_root=data_root,
            device=device,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        fold_accuracies.append(best_acc)
        all_histories[f"fold_{fold}"] = history

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print(f"\n{'='*60}")
    print(f"  {mode.upper()} RESULTS — {num_folds}-Fold Cross Validation")
    print(f"{'='*60}")
    for i, acc in enumerate(fold_accuracies, 1):
        print(f"  Fold {i}: {acc:.1f}%")
    print(f"  ─────────────────────")
    print(f"  Mean:   {mean_acc:.1f}% ± {std_acc:.1f}%")
    print(f"{'='*60}")

    os.makedirs("results", exist_ok=True)
    results = {
        "dataset": dataset,
        "mode": mode,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "histories": all_histories,
    }
    result_prefix = f"{dataset}_{mode}" if dataset != "esc50" else mode
    with open(f"results/{result_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return mean_acc, std_acc


def main():
    parser = argparse.ArgumentParser(description="Helix Validation Experiment")
    parser.add_argument("--mode", type=str, default="raw",
                        choices=["raw", "spectrogram", "helix-raw", "helix-spectrogram", "attention-raw", "attention-spectrogram", "both"])
    parser.add_argument("--dataset", type=str, default="esc50",
                        choices=["esc50", "speechcommands"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = DATASET_CONFIG[args.dataset]["default_root"]

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("WARNING: Running on CPU — this will be slow!")

    if args.dataset != "speechcommands" and not os.path.exists(args.data_root):
        print(f"\nERROR: {args.dataset} not found at '{args.data_root}'")
        print("\nTo download:")
        print("  git clone https://github.com/karolpiczak/ESC-50.git data/ESC-50-master")
        print("\nOr download from Kaggle:")
        print("  https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50")
        return

    results = {}

    modes_to_run = []
    if args.mode == "both":
        modes_to_run = ["raw", "spectrogram"]
    else:
        modes_to_run = [args.mode]

    for mode in modes_to_run:
        mean, std = run_experiment(
            mode=mode,
            data_root=args.data_root,
            device=device,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        results[mode] = {"mean": mean, "std": std}

    if args.mode == "both":
        print(f"\n{'#'*60}")
        print(f"  COMPARISON: RAW WAVEFORM vs SPECTROGRAM ({args.dataset})")
        print(f"{'#'*60}")
        print(f"  Raw waveform:  {results['raw']['mean']:.1f}% ± {results['raw']['std']:.1f}%")
        print(f"  Spectrogram:   {results['spectrogram']['mean']:.1f}% ± {results['spectrogram']['std']:.1f}%")
        diff = results['raw']['mean'] - results['spectrogram']['mean']
        winner = "RAW WAVEFORM" if diff > 0 else "SPECTROGRAM"
        print(f"  Difference:    {abs(diff):.1f}% in favor of {winner}")
        print(f"{'#'*60}")


if __name__ == "__main__":
    main()
