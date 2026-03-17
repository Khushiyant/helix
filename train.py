import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datetime import datetime

try:
    import wandb
except ImportError:
    wandb = None

from model import RawWaveformMamba, SpectrogramMamba
from dataset import get_dataloaders, get_speechcommands_dataloaders, get_concat_speechcommands_dataloaders, get_urbansound8k_dataloaders, get_librispeech_dataloaders, get_librispeech_scaling_dataloaders, get_voxpopuli_scaling_dataloaders

N_LAYERS = 6

DATASET_CONFIG = {
    "esc50": {"num_classes": 50, "num_folds": 5, "default_root": "data/ESC-50-master"},
    "urbansound8k": {"num_classes": 10, "num_folds": 10, "default_root": "data/UrbanSound8K"},
    "speechcommands": {"num_classes": 35, "num_folds": 1, "default_root": "data"},
    "librispeech": {"num_classes": 921, "num_folds": 1, "default_root": "data/LibriSpeech"},
    "voxpopuli": {"num_classes": 1313, "num_folds": 1, "default_root": "data/VoxPopuli"},
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


def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=True, n_pool_tokens=None, scaler=None, grad_accum_steps=1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    use_amp = scaler is not None

    optimizer.zero_grad()
    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup(inputs, targets, alpha=0.3)
                outputs = model(inputs, n_pool_tokens=n_pool_tokens)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(inputs, n_pool_tokens=n_pool_tokens)
                loss = criterion(outputs, targets)

        scaled_loss = loss / grad_accum_steps

        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if use_mixup:
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        total += targets.size(0)
        total_loss += loss.item()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_pool_tokens=None, use_amp=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(inputs, n_pool_tokens=n_pool_tokens)
            loss = criterion(outputs, targets)

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def _ckpt_path(dataset, mode, fold, tag="best"):
    os.makedirs("checkpoints", exist_ok=True)
    prefix = f"{dataset}_{mode}" if dataset != "esc50" else mode
    return f"checkpoints/{prefix}_fold{fold}_{tag}.pt"


def _save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_acc, history):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_acc": best_acc,
        "history": history,
    }, path)


def _load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt["scaler_state_dict"] is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt["best_acc"], ckpt["history"]


def train_fold(fold, mode, data_root, device, dataset="esc50", epochs=100, batch_size=32, lr=3e-4, n_clips=1, n_pool_tokens=None, use_wandb=False, use_amp=False, grad_accum_steps=1, save_every=0, resume=False, target_seconds=0):
    cfg = DATASET_CONFIG[dataset]
    num_folds = cfg["num_folds"]
    num_classes = cfg["num_classes"]

    print(f"\n{'='*60}")
    print(f"  FOLD {fold}/{num_folds} — Mode: {mode} — Dataset: {dataset}" +
          (f" — {target_seconds}s clips" if target_seconds > 0 else "") +
          (f" — n_clips: {n_clips}" if n_clips > 1 else "") +
          (" — AMP" if use_amp else "") +
          (f" — grad_accum: {grad_accum_steps}" if grad_accum_steps > 1 else ""))
    print(f"{'='*60}")

    data_mode = mode.replace("helix-", "").replace("attention-", "")
    if dataset == "voxpopuli":
        clip_seconds = target_seconds if target_seconds > 0 else 30
        train_loader, test_loader, num_classes = get_voxpopuli_scaling_dataloaders(
            root=data_root, target_seconds=clip_seconds, batch_size=batch_size,
        )
    elif dataset == "librispeech" and target_seconds > 0:
        train_loader, test_loader, num_classes = get_librispeech_scaling_dataloaders(
            root=data_root, target_seconds=target_seconds, batch_size=batch_size,
        )
    elif dataset == "speechcommands" and n_clips > 1:
        train_loader, test_loader = get_concat_speechcommands_dataloaders(
            root=data_root, n_clips=n_clips, batch_size=batch_size,
        )
    elif dataset == "speechcommands":
        train_loader, test_loader = get_speechcommands_dataloaders(
            root=data_root, test_fold=fold, batch_size=batch_size, mode=data_mode,
        )
    elif dataset == "librispeech":
        train_loader, test_loader, num_classes = get_librispeech_dataloaders(
            root=data_root, test_fold=fold, batch_size=batch_size, mode=data_mode,
        )
    elif dataset == "urbansound8k":
        train_loader, test_loader = get_urbansound8k_dataloaders(
            root=data_root, test_fold=fold, batch_size=batch_size, mode=data_mode,
        )
    else:
        train_loader, test_loader = get_dataloaders(
            root=data_root, test_fold=fold, batch_size=batch_size, mode=data_mode,
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

    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=50)
        wandb.config.update({"param_count": param_count}, allow_val_change=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    if use_amp and device.type != "cuda":
        print("  WARNING: --amp requires CUDA, disabling AMP for this fold")
        use_amp = False
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_acc = 0
    start_epoch = 1
    history = []

    resume_path = _ckpt_path(dataset, mode, fold, tag="latest")
    if resume and os.path.exists(resume_path):
        start_epoch_loaded, best_acc, history = _load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device,
        )
        start_epoch = start_epoch_loaded + 1
        print(f"  Resumed from epoch {start_epoch_loaded} (best_acc={best_acc:.1f}%)")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            n_pool_tokens=n_pool_tokens,
            scaler=scaler,
            grad_accum_steps=grad_accum_steps,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                       n_pool_tokens=n_pool_tokens,
                                       use_amp=use_amp)
        scheduler.step()

        elapsed = time.time() - t0

        if test_acc > best_acc:
            best_acc = test_acc
            _save_checkpoint(
                _ckpt_path(dataset, mode, fold, tag="best"),
                model, optimizer, scheduler, scaler, epoch, best_acc, history,
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

        # Periodic + latest checkpoint for resume
        if save_every > 0 and epoch % save_every == 0:
            _save_checkpoint(
                _ckpt_path(dataset, mode, fold, tag=f"epoch{epoch}"),
                model, optimizer, scheduler, scaler, epoch, best_acc, history,
            )
        _save_checkpoint(
            _ckpt_path(dataset, mode, fold, tag="latest"),
            model, optimizer, scheduler, scaler, epoch, best_acc, history,
        )

        if use_wandb:
            try:
                wandb.log({
                    f"fold_{fold}/train_loss": train_loss,
                    f"fold_{fold}/train_acc": train_acc,
                    f"fold_{fold}/test_loss": test_loss,
                    f"fold_{fold}/test_acc": test_acc,
                    f"fold_{fold}/best_acc": best_acc,
                    f"fold_{fold}/lr": optimizer.param_groups[0]["lr"],
                    f"fold_{fold}/epoch_time": elapsed,
                    "epoch": epoch,
                })
            except Exception as e:
                print(f"  WARNING: wandb.log() failed: {e}")

        if epoch % 10 == 0 or epoch == start_epoch:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.4f} / {train_acc:.1f}% | "
                f"Test: {test_loss:.4f} / {test_acc:.1f}% | "
                f"Best: {best_acc:.1f}% | "
                f"{elapsed:.1f}s"
            )

    if use_wandb:
        wandb.summary[f"fold_{fold}/best_acc"] = best_acc
        best_ckpt = _ckpt_path(dataset, mode, fold, tag="best")
        if os.path.exists(best_ckpt):
            artifact = wandb.Artifact(
                f"{dataset}_{mode}_fold{fold}_best", type="model",
                metadata={"fold": fold, "best_acc": best_acc},
            )
            artifact.add_file(best_ckpt)
            wandb.log_artifact(artifact)
        wandb.unwatch(model)

    print(f"\n  Fold {fold} best accuracy: {best_acc:.1f}%")
    return best_acc, history


def run_experiment(mode, data_root, device, dataset="esc50", epochs=100, batch_size=32, lr=3e-4, n_clips=1, n_pool_tokens=None, use_wandb=False, wandb_project=None, wandb_entity=None, use_amp=False, grad_accum_steps=1, save_every=0, resume=False, target_seconds=0):
    cfg = DATASET_CONFIG[dataset]
    num_folds = cfg["num_folds"]

    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT: {mode.upper()} — Dataset: {dataset}")
    if target_seconds > 0:
        print(f"  Scaling: {target_seconds}s clips, speaker ID")
    if n_clips > 1:
        print(f"  Long-seq: n_clips={n_clips}, total_audio={n_clips}s, total_tokens={n_clips * 100}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    if use_amp:
        print(f"  AMP: enabled, Grad accum: {grad_accum_steps}")
    print(f"{'#'*60}")

    if use_wandb:
        if target_seconds > 0:
            run_name = f"{dataset}_{target_seconds}s_{mode}"
        elif n_clips > 1:
            run_name = f"longseq_n{n_clips}_{mode}"
        else:
            run_name = f"{dataset}_{mode}"
        try:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                config={
                    "dataset": dataset,
                    "mode": mode,
                    "num_classes": cfg["num_classes"],
                    "n_clips": n_clips,
                    "n_pool_tokens": n_pool_tokens,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": 0.05,
                    "mixup_alpha": 0.3,
                    "grad_clip": 1.0,
                    "lr_min": 1e-6,
                    "n_layers": N_LAYERS,
                    "num_folds": num_folds,
                    "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                    "gpu_vram_gb": round(torch.cuda.get_device_properties(device).total_memory / 1e9, 1) if device.type == "cuda" else 0,
                    "use_amp": use_amp,
                    "grad_accum_steps": grad_accum_steps,
                    "save_every": save_every,
                    "resume": resume,
                    "target_seconds": target_seconds,
                },
            )
        except Exception as e:
            print(f"WARNING: wandb.init() failed: {e}. Disabling wandb for this run.")
            use_wandb = False

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
            n_clips=n_clips,
            n_pool_tokens=n_pool_tokens,
            use_wandb=use_wandb,
            use_amp=use_amp,
            grad_accum_steps=grad_accum_steps,
            save_every=save_every,
            resume=resume,
            target_seconds=target_seconds,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "dataset": dataset,
        "mode": mode,
        "n_clips": n_clips,
        "total_audio_seconds": n_clips,  # each Speech Commands clip is exactly 1 s
        "total_tokens": n_clips * 100,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "histories": all_histories,
    }
    if target_seconds > 0:
        result_file = f"results/{dataset}_{target_seconds}s_{mode}_{timestamp}.json"
        results["target_seconds"] = target_seconds
    elif n_clips > 1:
        result_file = f"results/longseq_n{n_clips}_{mode}_{timestamp}.json"
    else:
        result_prefix = f"{dataset}_{mode}" if dataset != "esc50" else mode
        result_file = f"results/{result_prefix}_{timestamp}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {result_file}")

    if use_wandb:
        wandb.summary["mean_accuracy"] = mean_acc
        wandb.summary["std_accuracy"] = std_acc
        for i, acc in enumerate(fold_accuracies, 1):
            wandb.summary[f"fold_{i}/final_acc"] = acc
        wandb.save(os.path.abspath(result_file), base_path=os.path.abspath("results"))
        wandb.finish()

    return mean_acc, std_acc


def main():
    parser = argparse.ArgumentParser(description="Helix Validation Experiment")
    parser.add_argument("--mode", type=str, default="raw",
                        choices=["raw", "spectrogram", "helix-raw", "helix-spectrogram", "attention-raw", "attention-spectrogram", "both"])
    parser.add_argument("--dataset", type=str, default="esc50",
                        choices=["esc50", "urbansound8k", "speechcommands", "librispeech", "voxpopuli"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_clips", type=int, default=1,
                        help="Number of clips to concatenate (long-seq experiment)")
    parser.add_argument("--target_seconds", type=int, default=0,
                        help="Clip duration for scaling experiment (0=disabled, e.g. 30, 60, 300)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision training (AMP)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save periodic checkpoint every N epochs (0=disabled, latest always saved)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="helix")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    use_wandb = args.wandb
    if use_wandb and wandb is None:
        print("WARNING: --wandb flag set but wandb is not installed. Disabling.")
        use_wandb = False

    if args.data_root is None:
        args.data_root = DATASET_CONFIG[args.dataset]["default_root"]

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("WARNING: Running on CPU — this will be slow!")

    if args.dataset not in ("speechcommands", "librispeech", "voxpopuli") and not os.path.exists(args.data_root):
        print(f"\nERROR: {args.dataset} not found at '{args.data_root}'")
        if args.dataset == "urbansound8k":
            print("\nDownload UrbanSound8K from:")
            print("  https://urbansounddataset.weebly.com/urbansound8k.html")
            print("\nExtract so the structure is: data/UrbanSound8K/{audio/,metadata/}")
        else:
            print("\nTo download:")
            print("  git clone https://github.com/karolpiczak/ESC-50.git data/ESC-50-master")
            print("\nOr download from Kaggle:")
            print("  https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50")
        return

    n_pool_tokens = 100 if args.n_clips > 1 else None

    results = {}

    modes_to_run = ["raw", "spectrogram"] if args.mode == "both" else [args.mode]

    for mode in modes_to_run:
        mean, std = run_experiment(
            mode=mode,
            data_root=args.data_root,
            device=device,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_clips=args.n_clips,
            n_pool_tokens=n_pool_tokens,
            use_wandb=use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            use_amp=args.amp,
            grad_accum_steps=args.grad_accum,
            save_every=args.save_every,
            resume=args.resume,
            target_seconds=args.target_seconds,
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
