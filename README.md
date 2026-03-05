# Helix
---

## Setup

```bash
uv sync
```

Download ESC-50:

```bash
git clone https://github.com/karolpiczak/ESC-50.git data/ESC-50-master
```

## Usage

```bash
# Train raw waveform model
python train.py --mode raw --data_root data/ESC-50-master

# Train spectrogram model
python train.py --mode spectrogram --data_root data/ESC-50-master

# Train both and compare
python train.py --mode both --data_root data/ESC-50-master
```

## Test models

```bash
python main.py
```
