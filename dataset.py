import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader


class ESC50Raw(Dataset):

    def __init__(self, root, fold_list, augment=False):
        self.audio_dir = os.path.join(root, "audio")
        self.augment = augment
        self.target_sr = 16000
        self.target_length = 80000

        meta = pd.read_csv(os.path.join(root, "meta", "esc50.csv"))
        self.data = meta[meta["fold"].isin(fold_list)].reset_index(drop=True)

        print(f"  Loaded {len(self.data)} clips from folds {fold_list}")

    def __len__(self):
        return len(self.data)

    def _load_and_resample(self, path):
        waveform, sr = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        if waveform.shape[1] < self.target_length:
            pad = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]

        return waveform

    def _augment(self, waveform):
        shift = np.random.randint(-8000, 8000)
        waveform = torch.roll(waveform, shift, dims=-1)

        scale = np.random.uniform(0.8, 1.2)
        waveform = waveform * scale

        noise = torch.randn_like(waveform) * 0.005
        waveform = waveform + noise

        return waveform

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(self.audio_dir, row["filename"])
        label = row["target"]

        waveform = self._load_and_resample(path)

        if self.augment:
            waveform = self._augment(waveform)

        return waveform, label


class ESC50Spectrogram(Dataset):

    def __init__(self, root, fold_list, augment=False):
        self.audio_dir = os.path.join(root, "audio")
        self.augment = augment
        self.target_sr = 16000
        self.target_length = 80000

        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        meta = pd.read_csv(os.path.join(root, "meta", "esc50.csv"))
        self.data = meta[meta["fold"].isin(fold_list)].reset_index(drop=True)

        print(f"  Loaded {len(self.data)} clips from folds {fold_list}")

    def __len__(self):
        return len(self.data)

    def _load_and_resample(self, path):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        if waveform.shape[1] < self.target_length:
            pad = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]
        return waveform

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(self.audio_dir, row["filename"])
        label = row["target"]

        waveform = self._load_and_resample(path)

        mel = self.mel_transform(waveform)
        mel = self.amplitude_to_db(mel)

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return mel, label


def get_dataloaders(root, test_fold, batch_size=32, mode="raw"):
    all_folds = [1, 2, 3, 4, 5]
    train_folds = [f for f in all_folds if f != test_fold]

    DatasetClass = ESC50Raw if mode == "raw" else ESC50Spectrogram

    print(f"\nCreating {mode} dataloaders (test fold: {test_fold})")
    train_dataset = DatasetClass(root, train_folds, augment=True)
    test_dataset = DatasetClass(root, [test_fold], augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/ESC-50-master"

    if not os.path.exists(root):
        print(f"ESC-50 not found at {root}")
        print("Download from: https://github.com/karolpiczak/ESC-50")
        print("Usage: python dataset.py /path/to/ESC-50-master")
        exit(1)

    train_loader, test_loader = get_dataloaders(root, test_fold=5, mode="raw")
    batch = next(iter(train_loader))
    print(f"Raw waveform batch: input={batch[0].shape}, labels={batch[1].shape}")

    train_loader, test_loader = get_dataloaders(root, test_fold=5, mode="spectrogram")
    batch = next(iter(train_loader))
    print(f"Spectrogram batch:  input={batch[0].shape}, labels={batch[1].shape}")
