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
