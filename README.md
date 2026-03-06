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

### ESC-50

```bash
python train.py --mode raw --dataset esc50
python train.py --mode spectrogram --dataset esc50
python train.py --mode helix-raw --dataset esc50
python train.py --mode attention-raw --dataset esc50
python train.py --mode both --dataset esc50
```

### Speech Commands

Speech Commands v2 is downloaded automatically on first run.

```bash
python train.py --dataset speechcommands --mode raw
python train.py --dataset speechcommands --mode helix-raw
python train.py --dataset speechcommands --mode attention-raw
```

### Long-Sequence Experiment

Concatenates N clips end-to-end; task is to classify the first clip (tests long-range memory). Only the first 100 tokens are pooled for classification.

```bash
python train.py --dataset speechcommands --mode raw --n_clips 10 --epochs 50
python train.py --dataset speechcommands --mode helix-raw --n_clips 10 --epochs 50
python train.py --dataset speechcommands --mode attention-raw --n_clips 10 --epochs 50
# Sweep longer sequences
python train.py --dataset speechcommands --mode raw --n_clips 50
python train.py --dataset speechcommands --mode raw --n_clips 100
```

### Results

Results are saved to `results/` with timestamps to prevent overwriting:
- Standard runs: `results/{dataset}_{mode}_{timestamp}.json`
- Long-seq runs: `results/longseq_n{N}_{mode}_{timestamp}.json`

## Test models

```bash
python main.py
```


## If you having problem in setting up Mamba
```bash
#!/bin/bash
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDAHOSTCXX=/usr/bin/g++-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
pip install mamba-ssm soundfile
```
