import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import soundfile as sf


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
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # shape: (channels, time)
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
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


SPEECH_COMMANDS_LABELS = sorted([
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
    'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual',
    'wow', 'yes', 'zero',
])

SPEECH_COMMANDS_LABEL_TO_IDX = {label: i for i, label in enumerate(SPEECH_COMMANDS_LABELS)}

_SC_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
_SC_FOLDER = "SpeechCommands/speech_commands_v0.02"


def _download_speech_commands(root):
    """Download and extract Speech Commands v2 if not already present."""
    import tarfile
    import urllib.request

    data_dir = os.path.join(root, _SC_FOLDER)
    if os.path.isdir(data_dir) and os.listdir(data_dir):
        return data_dir

    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(root, "speech_commands_v0.02.tar.gz")

    if not os.path.exists(tar_path):
        print(f"  Downloading Speech Commands v2 (~2.3GB)...")
        urllib.request.urlretrieve(_SC_URL, tar_path)

    print(f"  Extracting to {data_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)

    os.remove(tar_path)
    return data_dir


def _load_speech_commands_split(root, subset):
    """Build file list for a given split (training/validation/testing)."""
    data_dir = _download_speech_commands(root)

    val_file = os.path.join(data_dir, "validation_list.txt")
    test_file = os.path.join(data_dir, "testing_list.txt")

    with open(val_file) as f:
        val_set = set(f.read().strip().splitlines())
    with open(test_file) as f:
        test_set = set(f.read().strip().splitlines())

    label_set = set(SPEECH_COMMANDS_LABELS)
    all_files = []
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir) or label not in label_set:
            continue
        for fname in os.listdir(label_dir):
            if not fname.endswith(".wav"):
                continue
            rel_path = f"{label}/{fname}"
            if subset == "validation" and rel_path in val_set:
                all_files.append((os.path.join(label_dir, fname), label))
            elif subset == "testing" and rel_path in test_set:
                all_files.append((os.path.join(label_dir, fname), label))
            elif subset == "training" and rel_path not in val_set and rel_path not in test_set:
                all_files.append((os.path.join(label_dir, fname), label))

    return all_files


class SpeechCommandsRaw(Dataset):

    def __init__(self, root, subset="training", augment=False):
        self.files = _load_speech_commands_split(root, subset)
        self.augment = augment
        self.target_sr = 16000
        self.target_length = 16000  # 1 second at 16kHz

        print(f"  Loaded {len(self.files)} clips ({subset})")

    def __len__(self):
        return len(self.files)

    def _augment(self, waveform):
        shift = np.random.randint(-1600, 1600)
        waveform = torch.roll(waveform, shift, dims=-1)

        scale = np.random.uniform(0.8, 1.2)
        waveform = waveform * scale

        noise = torch.randn_like(waveform) * 0.005
        waveform = waveform + noise

        return waveform

    def __getitem__(self, idx):
        path, label = self.files[idx]
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = T.Resample(sr, self.target_sr)(waveform)

        if waveform.shape[1] < self.target_length:
            pad = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]

        if self.augment:
            waveform = self._augment(waveform)

        return waveform, SPEECH_COMMANDS_LABEL_TO_IDX[label]


class ConcatSpeechCommandsRaw(Dataset):
    """Concatenates N Speech Commands clips along time axis.
    Label = first clip's class (long-range memory test).
    """

    def __init__(self, root, subset="training", augment=False, n_clips=10):
        self.inner = SpeechCommandsRaw(root, subset=subset, augment=False)
        self.augment = augment
        self.n_clips = n_clips
        self.deterministic = subset != "training"

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        waveform, label = self.inner[idx]

        if self.n_clips > 1:
            parts = [waveform]
            n = len(self.inner)
            # Per-idx seed: worker-safe and reproducible for val/test
            rng = np.random.RandomState(idx) if self.deterministic else np.random
            for _ in range(self.n_clips - 1):
                filler_idx = rng.randint(0, n)
                filler_wav, _ = self.inner[filler_idx]
                parts.append(filler_wav)
            waveform = torch.cat(parts, dim=-1)  # (1, n_clips * 16000)

        if self.augment:
            waveform = self.inner._augment(waveform)

        return waveform, label


class SpeechCommandsSpectrogram(Dataset):

    def __init__(self, root, subset="training", augment=False):
        self.files = _load_speech_commands_split(root, subset)
        self.augment = augment
        self.target_sr = 16000
        self.target_length = 16000

        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        print(f"  Loaded {len(self.files)} clips ({subset})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = T.Resample(sr, self.target_sr)(waveform)

        if waveform.shape[1] < self.target_length:
            pad = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]

        mel = self.mel_transform(waveform)
        mel = self.amplitude_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return mel, SPEECH_COMMANDS_LABEL_TO_IDX[label]


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


def get_speechcommands_dataloaders(root, test_fold=1, batch_size=32, mode="raw"):  # noqa: ARG001
    DatasetClass = SpeechCommandsRaw if mode == "raw" else SpeechCommandsSpectrogram

    print(f"\nCreating Speech Commands {mode} dataloaders")
    train_dataset = DatasetClass(root, subset="training", augment=True)
    test_dataset = DatasetClass(root, subset="testing", augment=False)

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


def get_concat_speechcommands_dataloaders(root, n_clips=10, batch_size=32):
    print(f"\nCreating Concat Speech Commands dataloaders (n_clips={n_clips})")
    train_dataset = ConcatSpeechCommandsRaw(root, subset="training", augment=True, n_clips=n_clips)
    test_dataset = ConcatSpeechCommandsRaw(root, subset="testing", augment=False, n_clips=n_clips)

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

    dataset = sys.argv[1] if len(sys.argv) > 1 else "esc50"

    if dataset == "speechcommands":
        root = sys.argv[2] if len(sys.argv) > 2 else "data"

        train_loader, test_loader = get_speechcommands_dataloaders(root, mode="raw")
        batch = next(iter(train_loader))
        print(f"Raw waveform batch: input={batch[0].shape}, labels={batch[1].shape}")

        train_loader, test_loader = get_speechcommands_dataloaders(root, mode="spectrogram")
        batch = next(iter(train_loader))
        print(f"Spectrogram batch:  input={batch[0].shape}, labels={batch[1].shape}")
    else:
        root = sys.argv[2] if len(sys.argv) > 2 else "data/ESC-50-master"
        if not os.path.exists(root):
            print(f"ESC-50 not found at {root}")
            print("Download from: https://github.com/karolpiczak/ESC-50")
            exit(1)

        train_loader, test_loader = get_dataloaders(root, test_fold=5, mode="raw")
        batch = next(iter(train_loader))
        print(f"Raw waveform batch: input={batch[0].shape}, labels={batch[1].shape}")

        train_loader, test_loader = get_dataloaders(root, test_fold=5, mode="spectrogram")
        batch = next(iter(train_loader))
        print(f"Spectrogram batch:  input={batch[0].shape}, labels={batch[1].shape}")
