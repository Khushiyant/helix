# Helix

A hybrid audio classification architecture combining Bidirectional Mamba (state space models) with self-attention layers. Helix enables fair, controlled comparisons between Pure Mamba, Pure Attention, and Mamba+Attention hybrid architectures on audio classification tasks.

## Architecture

Helix provides three model variants, all parameter-matched (~8.3M params) for fair comparison:

| Variant | Layers | Description |
|---|---|---|
| **Pure Mamba** | 6x BiMamba | Bidirectional Mamba blocks (forward + backward SSM) |
| **HELIX** | 5x BiMamba + 1x Attention | Hybrid — attention inserted at layer 3 |
| **Pure Attention** | 6x Attention | Multi-head self-attention with FFN |

Each variant supports two input representations:

- **Raw waveform** — 1D conv patch embedding (kernel=160, stride=160) over 16kHz audio
- **Spectrogram** — 2D conv patch embedding (16x16 patches) over 128-bin mel-spectrogram

The `SelfAttentionBlock` FFN dimension is computed dynamically to match the parameter count of `BiMambaBlock`, ensuring all variants have identical capacity.

### Model parameters

| Parameter | Value |
|---|---|
| `d_model` | 256 |
| `d_state` | 32 |
| `d_conv` | 4 |
| `expand` | 2 |
| `n_layers` | 6 |
| `num_heads` | 4 |

## Datasets

### ESC-50

Environmental Sound Classification — 2,000 five-second clips across 50 classes. Uses 5-fold cross-validation. Must be downloaded manually:

```bash
git clone https://github.com/karolpiczak/ESC-50.git data/ESC-50-master
```

### Speech Commands v2

~105k one-second spoken command clips across 35 classes. Auto-downloads (~2.3GB) on first run. Uses the official train/validation/test split.

### Concatenated Speech Commands

Chains N clips end-to-end for long-range memory testing. The label is the first clip's class, and only the first 100 tokens are pooled for classification. Tests whether the model retains information over long sequences.

## Setup

### Requirements

- Python >= 3.10
- CUDA-compatible GPU (recommended)

### Install

```bash
# Core dependencies
uv sync

# With wandb logging support
pip install -e ".[wandb]"
```

### Mamba setup (if having CUDA issues)

```bash
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDAHOSTCXX=/usr/bin/g++-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
pip install mamba-ssm soundfile
```

## Usage

### Training

```bash
python train.py --dataset <dataset> --mode <mode> [options]
```

**Modes:**

| Mode | Architecture | Input |
|---|---|---|
| `raw` | Pure Mamba | Raw waveform |
| `spectrogram` | Pure Mamba | Mel-spectrogram |
| `helix-raw` | HELIX hybrid | Raw waveform |
| `helix-spectrogram` | HELIX hybrid | Mel-spectrogram |
| `attention-raw` | Pure Attention | Raw waveform |
| `attention-spectrogram` | Pure Attention | Mel-spectrogram |
| `both` | Pure Mamba | Runs `raw` and `spectrogram` back-to-back |

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `esc50` | `esc50` or `speechcommands` |
| `--data_root` | auto | Override dataset path |
| `--epochs` | `100` | Training epochs per fold |
| `--batch_size` | `32` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--n_clips` | `1` | Clips to concatenate (long-sequence experiment) |
| `--gpu` | `0` | GPU device index |
| `--wandb` | off | Enable Weights & Biases logging |
| `--wandb_project` | `helix` | W&B project name |
| `--wandb_entity` | none | W&B team/user entity |

### Examples

```bash
# ESC-50 experiments
python train.py --dataset esc50 --mode raw
python train.py --dataset esc50 --mode helix-raw
python train.py --dataset esc50 --mode attention-raw

# Speech Commands experiments
python train.py --dataset speechcommands --mode raw
python train.py --dataset speechcommands --mode helix-raw

# Long-sequence memory test
python train.py --dataset speechcommands --mode raw --n_clips 10
python train.py --dataset speechcommands --mode helix-raw --n_clips 50

# Compare raw vs spectrogram
python train.py --dataset esc50 --mode both

# With wandb logging
python train.py --dataset esc50 --mode helix-raw --wandb --wandb_project my-project
```

### Training details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Weight decay | 0.05 |
| LR schedule | Cosine annealing (eta_min=1e-6) |
| Gradient clipping | max_norm=1.0 |
| Mixup | Beta(0.3, 0.3) |
| Loss | Cross-entropy |

Data augmentation (training only):
- Random time shift
- Random amplitude scaling (0.8x-1.2x)
- Gaussian noise injection (sigma=0.005)

## Weights & Biases integration

Enable with the `--wandb` flag. Requires `pip install -e ".[wandb]"` and a wandb login:

```bash
wandb login
python train.py --dataset esc50 --mode helix-raw --wandb
```

**What gets logged:**

- **Config** — all hyperparameters (dataset, mode, epochs, lr, batch_size, etc.)
- **Per-epoch metrics** — `fold_N/train_loss`, `train_acc`, `test_loss`, `test_acc`, `best_acc`, `lr`, `epoch_time`
- **Summary** — `mean_accuracy`, `std_accuracy`, per-fold `fold_N/final_acc`
- **Gradients** — gradient histograms via `wandb.watch()` (logged every 50 batches)
- **Artifacts** — result JSON file uploaded to the run

Each `run_experiment()` call creates one W&B run. Using `--mode both` creates two separate runs (one per input type).

wandb is treated as a non-critical sidecar — if `wandb.init()` or `wandb.log()` fails (e.g. network issues), training continues with a warning.

## Experiment Results

All experiments ran on a single **NVIDIA L40S** GPU (48GB), Python 3.10, CUDA 12.7. Training used 100 epochs with cosine LR schedule (lr=3e-4, min=1e-6), AdamW (weight_decay=0.05), mixup (alpha=0.3), and gradient clipping (max_norm=1.0).

W&B project: [`helix`](https://wandb.ai/khushiyant-personal/helix) (15 total runs — 8 finished, 7 crashed due to timeout)

### ESC-50 (5-fold cross-validation)

All 6 ESC-50 runs completed successfully (100 epochs each).

| Mode | Architecture | Input | Mean Best Acc (%) | Mean Test Acc (%) | Runtime |
|---|---|---|---|---|---|
| `raw` | Pure Mamba | Raw waveform | **55.10** | **53.50** | ~4.2h |
| `spectrogram` | Pure Mamba | Spectrogram | 53.75 | 52.75 | ~3.1h |
| `helix-raw` | HELIX Hybrid | Raw waveform | 50.20 | 49.55 | ~4.1h |
| `helix-spectrogram` | HELIX Hybrid | Spectrogram | 51.85 | 49.90 | ~2.9h |
| `attention-raw` | Pure Attention | Raw waveform | 44.60 | 43.20 | ~3.6h |
| `attention-spectrogram` | Pure Attention | Spectrogram | 46.10 | 44.65 | ~2.4h |

<details>
<summary>Per-fold breakdown (Best Accuracy %)</summary>

| Mode | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|
| `raw` | 52.25 | 51.50 | 56.50 | **60.75** | 54.50 |
| `spectrogram` | 50.25 | 52.50 | 55.50 | **58.50** | 52.00 |
| `helix-raw` | 48.50 | 44.00 | 53.75 | **54.00** | 50.75 |
| `helix-spectrogram` | 49.00 | 52.00 | 51.25 | **56.25** | 50.75 |
| `attention-raw` | 45.25 | 40.50 | 42.75 | **50.00** | 44.50 |
| `attention-spectrogram` | 44.00 | 42.50 | 44.75 | **49.50** | 49.75 |

</details>

**Key findings (ESC-50):**
- **Pure Mamba dominates** across both input types, outperforming both Helix and Attention
- **Raw waveform > Spectrogram** for Mamba-based models (55.10% vs 53.75%)
- **Pure Attention struggles** on ESC-50, ~10% behind Pure Mamba
- Fold 4 consistently yields the highest accuracy across all architectures

### Speech Commands v2 (single fold)

Most Speech Commands runs crashed due to compute timeout (~12.8h limit) before completing 100 epochs. Crashed runs still logged partial metrics up to the crash point.

| Mode | Architecture | Input | Best Acc (%) | Test Acc (%) | Epochs | Status |
|---|---|---|---|---|---|---|
| `helix-raw` | HELIX Hybrid | Raw waveform | **92.94** | **92.79** | 81/100 | crashed |
| `helix-spectrogram` | HELIX Hybrid | Spectrogram | 92.44 | 92.44 | 94/100 | crashed |
| `spectrogram` | Pure Mamba | Spectrogram | 92.36 | 92.36 | 86/100 | crashed |
| `attention-spectrogram` | Pure Attention | Spectrogram | 91.27 | 91.26 | 100/100 | finished |
| `raw` | Pure Mamba | Raw waveform | 90.27 | 54.54 | 76/100 | crashed |
| `attention-raw` | Pure Attention | Raw waveform | 82.43 | 78.86 | 100/100 | finished |
| `attention-raw` (dup) | Pure Attention | Raw waveform | 81.83 | 77.02 | 100/100 | finished |

**Key findings (Speech Commands):**
- **HELIX hybrid achieves the highest accuracy** (92.94%) even without completing training, suggesting the hybrid Mamba+Attention design excels on temporal speech patterns
- **Spectrogram input converges faster** — all spectrogram runs reached >91% within their allotted epochs
- **Pure Attention with raw waveform lags significantly** (~78-82%), confirming attention alone struggles with long 1D sequences
- The crashed `raw` Pure Mamba run shows a test accuracy of 54.54% at epoch 76 (possibly mid-training instability), despite a best_acc of 90.27%

### Long-Sequence Experiments (n_clips=10, Speech Commands)

Concatenates 10 clips end-to-end (~10 seconds of audio) to test long-range memory. Both runs crashed at epoch 24 due to the ~24min/epoch cost.

| Mode | Architecture | Best Acc (%) | Test Acc (%) | Epochs | Epoch Time |
|---|---|---|---|---|---|
| `helix-raw` | HELIX Hybrid | **91.31** | **90.44** | 24/100 | ~24.3 min |
| `raw` | Pure Mamba | 89.81 | 89.36 | 24/100 | ~24.0 min |

**Key findings (Long-Sequence):**
- **HELIX outperforms Pure Mamba** by ~1.5% on the long-sequence task after just 24 epochs
- Both architectures achieve >89% accuracy even at 24 epochs, demonstrating strong long-range memory
- The attention layer in HELIX appears to provide a modest but consistent advantage for retaining information across long sequences

### Summary

| Ranking | ESC-50 (Mean Best Acc) | Speech Commands (Best Acc) | Long-Seq n=10 (Best Acc) |
|---|---|---|---|
| 1st | Pure Mamba Raw (55.10%) | HELIX Raw (92.94%) | HELIX Raw (91.31%) |
| 2nd | Pure Mamba Spec (53.75%) | HELIX Spec (92.44%) | Pure Mamba Raw (89.81%) |
| 3rd | HELIX Spec (51.85%) | Pure Mamba Spec (92.36%) | — |

- **Pure Mamba** is best for short environmental sounds (ESC-50)
- **HELIX Hybrid** excels on temporal/speech tasks and long sequences
- **Pure Attention** consistently underperforms, especially on raw waveform inputs

### Output files

Results are saved as JSON to `results/`:

- Standard: `results/{dataset}_{mode}_{timestamp}.json`
- Long-sequence: `results/longseq_n{N}_{mode}_{timestamp}.json`

Each file contains fold accuracies, mean/std, hyperparameters, and full per-epoch training histories.

### Plotting

```bash
python plot.py --results_dir results --modes raw helix-raw --out_dir plots
```

Generates three plots in `plots/`:
- `training_curves.png` — test loss and accuracy over epochs (mean +/- std across folds)
- `fold_comparison.png` — per-fold best accuracy bar chart
- `summary.png` — mean accuracy with error bars

### Verifying models

```bash
python model.py
```

Runs a forward pass through all model variants with dummy inputs, printing parameter counts and output shapes.

## Project structure

```
.
├── model.py       # BiMambaBlock, SelfAttentionBlock, RawWaveformMamba, SpectrogramMamba
├── dataset.py     # ESC-50, Speech Commands, ConcatSpeechCommands dataset classes
├── train.py       # Training loop, evaluation, experiment runner, wandb integration
├── plot.py        # Result visualization (training curves, fold comparison, summary)
├── pyproject.toml # Dependencies and project metadata
└── README.md
```
