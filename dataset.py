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
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
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
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)
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


class UrbanSound8KRaw(Dataset):

    def __init__(self, root, fold_list, augment=False):
        self.audio_dir = os.path.join(root, "audio")
        self.augment = augment
        self.target_sr = 16000
        self.target_length = 64000  # 4 seconds at 16kHz

        meta = pd.read_csv(os.path.join(root, "metadata", "UrbanSound8K.csv"))
        self.data = meta[meta["fold"].isin(fold_list)].reset_index(drop=True)

        print(f"  Loaded {len(self.data)} clips from folds {fold_list}")

    def __len__(self):
        return len(self.data)

    def _load_and_resample(self, path):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
        if waveform.shape[1] < self.target_length:
            pad = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]
        return waveform

    def _augment(self, waveform):
        shift = np.random.randint(-6400, 6400)
        waveform = torch.roll(waveform, shift, dims=-1)

        scale = np.random.uniform(0.8, 1.2)
        waveform = waveform * scale

        noise = torch.randn_like(waveform) * 0.005
        waveform = waveform + noise

        return waveform

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(self.audio_dir, f"fold{row['fold']}", row["slice_file_name"])
        label = row["classID"]

        waveform = self._load_and_resample(path)

        if self.augment:
            waveform = self._augment(waveform)

        return waveform, label


class UrbanSound8KSpectrogram(Dataset):

    def __init__(self, root, fold_list, augment=False):
        self.audio_dir = os.path.join(root, "audio")
        self.augment = augment
        self.target_sr = 16000
        self.target_length = 64000

        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        meta = pd.read_csv(os.path.join(root, "metadata", "UrbanSound8K.csv"))
        self.data = meta[meta["fold"].isin(fold_list)].reset_index(drop=True)

        print(f"  Loaded {len(self.data)} clips from folds {fold_list}")

    def __len__(self):
        return len(self.data)

    def _load_and_resample(self, path):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)
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
        path = os.path.join(self.audio_dir, f"fold{row['fold']}", row["slice_file_name"])
        label = row["classID"]

        waveform = self._load_and_resample(path)

        mel = self.mel_transform(waveform)
        mel = self.amplitude_to_db(mel)

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return mel, label


def get_urbansound8k_dataloaders(root, test_fold, batch_size=32, mode="raw"):
    all_folds = list(range(1, 11))
    train_folds = [f for f in all_folds if f != test_fold]

    DatasetClass = UrbanSound8KRaw if mode == "raw" else UrbanSound8KSpectrogram

    print(f"\nCreating UrbanSound8K {mode} dataloaders (test fold: {test_fold})")
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


def _discover_librispeech_files(root, subset):
    """Walk LibriSpeech directory to find all FLAC files and speaker IDs."""
    subset_dir = os.path.join(root, subset)
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"LibriSpeech subset not found: {subset_dir}")

    files = []
    for speaker_id in sorted(os.listdir(subset_dir)):
        speaker_dir = os.path.join(subset_dir, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue
        for chapter_id in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter_id)
            if not os.path.isdir(chapter_dir):
                continue
            for fname in os.listdir(chapter_dir):
                if fname.endswith(".flac"):
                    files.append((os.path.join(chapter_dir, fname), speaker_id))

    speakers = sorted(set(s for _, s in files))
    speaker_to_idx = {s: i for i, s in enumerate(speakers)}
    return files, speaker_to_idx


def _split_librispeech_by_utterance(root, subset="train-clean-100", val_ratio=0.2, seed=42):
    """Split a single LibriSpeech subset into train/val by utterance per speaker.

    Every speaker appears in both splits so the classifier sees all speakers
    during training. Utterances are held out, not speakers.
    """
    files, speaker_to_idx = _discover_librispeech_files(root, subset)

    speaker_files = {}
    for path, speaker_id in files:
        speaker_files.setdefault(speaker_id, []).append((path, speaker_id))

    rng = np.random.RandomState(seed)
    train_files = []
    val_files = []
    for speaker_id in sorted(speaker_files.keys()):
        utterances = speaker_files[speaker_id]
        rng.shuffle(utterances)
        n_val = max(1, int(len(utterances) * val_ratio))
        val_files.extend(utterances[:n_val])
        train_files.extend(utterances[n_val:])

    return train_files, val_files, speaker_to_idx


class LibriSpeechRaw(Dataset):

    def __init__(self, files, speaker_to_idx, augment=False):
        self.files = files
        self.speaker_to_idx = speaker_to_idx
        self.augment = augment
        self.target_sr = 16000
        self.target_length = 160000  # 10 seconds at 16kHz

        n_speakers = len(set(s for _, s in files))
        print(f"  Loaded {len(self.files)} clips ({n_speakers} speakers)")

    def __len__(self):
        return len(self.files)

    def _augment(self, waveform):
        shift = np.random.randint(-16000, 16000)
        waveform = torch.roll(waveform, shift, dims=-1)

        scale = np.random.uniform(0.8, 1.2)
        waveform = waveform * scale

        noise = torch.randn_like(waveform) * 0.005
        waveform = waveform + noise

        return waveform

    def __getitem__(self, idx):
        path, speaker_id = self.files[idx]
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            return torch.zeros(1, self.target_length), self.speaker_to_idx[speaker_id]
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

        return waveform, self.speaker_to_idx[speaker_id]


class LibriSpeechSpectrogram(Dataset):

    def __init__(self, files, speaker_to_idx, augment=False):
        self.files = files
        self.speaker_to_idx = speaker_to_idx
        self.augment = augment
        self.target_sr = 16000
        self.target_length = 160000

        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        n_speakers = len(set(s for _, s in files))
        print(f"  Loaded {len(self.files)} clips ({n_speakers} speakers)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, speaker_id = self.files[idx]
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            waveform = torch.zeros(1, self.target_length)
            mel = self.mel_transform(waveform)
            mel = self.amplitude_to_db(mel)
            return mel, self.speaker_to_idx[speaker_id]
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

        return mel, self.speaker_to_idx[speaker_id]


def get_librispeech_dataloaders(root, test_fold=1, batch_size=32, mode="raw"):  # noqa: ARG001
    DatasetClass = LibriSpeechRaw if mode == "raw" else LibriSpeechSpectrogram

    print(f"\nCreating LibriSpeech {mode} dataloaders (speaker ID classification)")
    train_files, val_files, speaker_to_idx = _split_librispeech_by_utterance(root)
    train_dataset = DatasetClass(train_files, speaker_to_idx, augment=True)
    test_dataset = DatasetClass(val_files, speaker_to_idx, augment=False)

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


def _parse_tedlium_stm(stm_path):
    """Parse a single STM file into a list of (start, end, speaker_id) segments."""
    segments = []
    with open(stm_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            speaker_id = parts[2]
            if speaker_id == "inter_segment_gap":
                continue
            start = float(parts[3])
            end = float(parts[4])
            if end - start < 0.5:
                continue
            segments.append((start, end, speaker_id))
    return segments


_TEDLIUM_URL = "http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz"


def _download_tedlium(root):
    """Download and extract TEDLIUM release 3 if not already present."""
    import subprocess
    import tarfile

    data_dir = os.path.join(root, "data", "stm")
    if os.path.isdir(data_dir) and os.listdir(data_dir):
        return

    parent = os.path.dirname(root)
    os.makedirs(parent, exist_ok=True)
    tar_path = os.path.join(parent, "TEDLIUM_release-3.tgz")

    if not os.path.exists(tar_path):
        print(f"  Downloading TEDLIUM release 3 (~50GB)...")
        ret = subprocess.run(
            ["curl", "-L", "-o", tar_path, "--progress-bar", _TEDLIUM_URL],
            check=False,
        )
        if ret.returncode != 0:
            if os.path.exists(tar_path):
                os.remove(tar_path)
            raise RuntimeError(
                f"Download failed (curl exit {ret.returncode}). "
                f"Download manually:\n  curl -L -o {tar_path} {_TEDLIUM_URL}"
            )

    # Validate it's actually gzip before extracting
    with open(tar_path, "rb") as f:
        magic = f.read(2)
    if magic != b'\x1f\x8b':
        os.remove(tar_path)
        raise RuntimeError(
            f"Downloaded file is not a valid gzip archive (got {magic!r}). "
            f"Download manually:\n  curl -L -o {tar_path} {_TEDLIUM_URL}"
        )

    print(f"  Extracting to {root}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(parent)

    os.remove(tar_path)
    print(f"  TEDLIUM ready at {root}")


def _build_tedlium_index(root, subset="data"):
    """Build index of (sph_path, segments) for a TEDLium subset."""
    _download_tedlium(root)
    stm_dir = os.path.join(root, subset, "stm")
    sph_dir = os.path.join(root, subset, "sph")
    if not os.path.isdir(stm_dir):
        raise FileNotFoundError(f"TEDLium STM dir not found: {stm_dir}")

    speaker_segments = {}
    for stm_file in sorted(os.listdir(stm_dir)):
        if not stm_file.endswith(".stm"):
            continue
        talk_id = stm_file.replace(".stm", "")
        sph_path = os.path.join(sph_dir, f"{talk_id}.sph")
        if not os.path.exists(sph_path):
            continue
        for start, end, speaker_id in _parse_tedlium_stm(os.path.join(stm_dir, stm_file)):
            speaker_segments.setdefault(speaker_id, []).append((sph_path, start, end))

    speakers = sorted(speaker_segments.keys())
    speaker_to_idx = {s: i for i, s in enumerate(speakers)}
    return speaker_segments, speaker_to_idx


def _split_tedlium_by_utterance(speaker_segments, val_ratio=0.2, seed=42):
    """Hold out utterances per speaker so every speaker is seen during training."""
    rng = np.random.RandomState(seed)
    train_segs = {}
    val_segs = {}
    for speaker_id in sorted(speaker_segments.keys()):
        utterances = list(speaker_segments[speaker_id])
        rng.shuffle(utterances)
        n_val = max(1, int(len(utterances) * val_ratio))
        val_segs[speaker_id] = utterances[:n_val]
        train_segs[speaker_id] = utterances[n_val:]
    return train_segs, val_segs


class TEDLIUMRaw(Dataset):
    """TEDLium speaker ID dataset. Stitches segments from the same speaker
    to create clips of a target duration for scaling experiments."""

    def __init__(self, speaker_segments, speaker_to_idx, target_seconds=30, augment=False):
        self.speaker_to_idx = speaker_to_idx
        self.augment = augment
        self.target_sr = 16000
        self.target_length = int(target_seconds * self.target_sr)
        self.target_seconds = target_seconds

        self.samples = []
        for speaker_id in sorted(speaker_segments.keys()):
            segs = speaker_segments[speaker_id]
            total_dur = sum(end - start for _, start, end in segs)
            n_samples = max(1, int(total_dur / target_seconds))
            for i in range(n_samples):
                self.samples.append((speaker_id, i, n_samples))

        self.speaker_segments = speaker_segments
        n_speakers = len(speaker_segments)
        print(f"  Loaded {len(self.samples)} samples ({n_speakers} speakers, {target_seconds}s clips)")

    def __len__(self):
        return len(self.samples)

    def _load_segment(self, sph_path, start, end):
        frame_start = int(start * self.target_sr)
        num_frames = int((end - start) * self.target_sr)
        try:
            data, sr = sf.read(sph_path, start=frame_start, frames=num_frames,
                               dtype="float32", always_2d=True)
        except Exception:
            return torch.zeros(1, num_frames)
        waveform = torch.from_numpy(data.T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = T.Resample(sr, self.target_sr)(waveform)
        return waveform

    def _augment(self, waveform):
        shift = np.random.randint(-self.target_sr, self.target_sr)
        waveform = torch.roll(waveform, shift, dims=-1)
        scale = np.random.uniform(0.8, 1.2)
        waveform = waveform * scale
        noise = torch.randn_like(waveform) * 0.005
        waveform = waveform + noise
        return waveform

    def __getitem__(self, idx):
        speaker_id, sample_idx, n_speaker_samples = self.samples[idx]
        segs = self.speaker_segments[speaker_id]

        rng = np.random.RandomState(idx) if not self.augment else np.random
        order = list(range(len(segs)))
        rng.shuffle(order)
        start_seg = (sample_idx * len(segs) // n_speaker_samples) % len(segs)
        order = order[start_seg:] + order[:start_seg]

        parts = []
        collected = 0
        for seg_idx in order:
            if collected >= self.target_length:
                break
            sph_path, start, end = segs[seg_idx]
            chunk = self._load_segment(sph_path, start, end)
            parts.append(chunk)
            collected += chunk.shape[1]

        waveform = torch.cat(parts, dim=-1) if parts else torch.zeros(1, self.target_length)

        if waveform.shape[1] < self.target_length:
            pad = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]

        if self.augment:
            waveform = self._augment(waveform)

        return waveform, self.speaker_to_idx[speaker_id]


class TEDLIUMSpectrogram(Dataset):

    def __init__(self, speaker_segments, speaker_to_idx, target_seconds=30, augment=False):
        self.speaker_to_idx = speaker_to_idx
        self.augment = augment
        self.target_sr = 16000
        self.target_length = int(target_seconds * self.target_sr)
        self.target_seconds = target_seconds

        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128, power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        self.samples = []
        for speaker_id in sorted(speaker_segments.keys()):
            segs = speaker_segments[speaker_id]
            total_dur = sum(end - start for _, start, end in segs)
            n_samples = max(1, int(total_dur / target_seconds))
            for i in range(n_samples):
                self.samples.append((speaker_id, i, n_samples))

        self.speaker_segments = speaker_segments
        n_speakers = len(speaker_segments)
        print(f"  Loaded {len(self.samples)} samples ({n_speakers} speakers, {target_seconds}s clips)")

    def __len__(self):
        return len(self.samples)

    def _load_segment(self, sph_path, start, end):
        frame_start = int(start * self.target_sr)
        num_frames = int((end - start) * self.target_sr)
        try:
            data, sr = sf.read(sph_path, start=frame_start, frames=num_frames,
                               dtype="float32", always_2d=True)
        except Exception:
            return torch.zeros(1, num_frames)
        waveform = torch.from_numpy(data.T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = T.Resample(sr, self.target_sr)(waveform)
        return waveform

    def __getitem__(self, idx):
        speaker_id, sample_idx, n_speaker_samples = self.samples[idx]
        segs = self.speaker_segments[speaker_id]

        rng = np.random.RandomState(idx) if not self.augment else np.random
        order = list(range(len(segs)))
        rng.shuffle(order)
        start_seg = (sample_idx * len(segs) // n_speaker_samples) % len(segs)
        order = order[start_seg:] + order[:start_seg]

        parts = []
        collected = 0
        for seg_idx in order:
            if collected >= self.target_length:
                break
            sph_path, start, end = segs[seg_idx]
            chunk = self._load_segment(sph_path, start, end)
            parts.append(chunk)
            collected += chunk.shape[1]

        waveform = torch.cat(parts, dim=-1) if parts else torch.zeros(1, self.target_length)

        if waveform.shape[1] < self.target_length:
            pad = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]

        mel = self.mel_transform(waveform)
        mel = self.amplitude_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return mel, self.speaker_to_idx[speaker_id]


def get_tedlium_dataloaders(root, test_fold=1, batch_size=32, mode="raw", target_seconds=30):  # noqa: ARG001
    DatasetClass = TEDLIUMRaw if mode == "raw" else TEDLIUMSpectrogram

    print(f"\nCreating TEDLIUM {mode} dataloaders ({target_seconds}s clips, speaker ID)")
    speaker_segments, speaker_to_idx = _build_tedlium_index(root)
    train_segs, val_segs = _split_tedlium_by_utterance(speaker_segments)

    train_dataset = DatasetClass(
        train_segs, speaker_to_idx,
        target_seconds=target_seconds, augment=True,
    )
    test_dataset = DatasetClass(
        val_segs, speaker_to_idx,
        target_seconds=target_seconds, augment=False,
    )

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

    return train_loader, test_loader, len(speaker_to_idx)


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

    if dataset == "tedlium":
        root = sys.argv[2] if len(sys.argv) > 2 else "data/TEDLIUM_release-3"
        if not os.path.exists(root):
            print(f"TEDLIUM not found at {root}")
            print("Download from: https://www.openslr.org/51/")
            exit(1)

        for secs in [30, 60]:
            train_loader, test_loader, n_speakers = get_tedlium_dataloaders(root, mode="raw", target_seconds=secs)
            batch = next(iter(train_loader))
            print(f"Raw {secs}s batch: input={batch[0].shape}, labels={batch[1].shape}, speakers={n_speakers}")
    elif dataset == "librispeech":
        root = sys.argv[2] if len(sys.argv) > 2 else "data/LibriSpeech"
        if not os.path.exists(root):
            print(f"LibriSpeech not found at {root}")
            print("Download from: https://www.openslr.org/12")
            exit(1)

        train_files, val_files, speaker_to_idx = _split_librispeech_by_utterance(root)
        print(f"Speakers: {len(speaker_to_idx)} total, "
              f"{len(set(s for _,s in train_files))} train, "
              f"{len(set(s for _,s in val_files))} val")

        train_loader, test_loader = get_librispeech_dataloaders(root, mode="raw")
        batch = next(iter(train_loader))
        print(f"Raw waveform batch: input={batch[0].shape}, labels={batch[1].shape}")

        train_loader, test_loader = get_librispeech_dataloaders(root, mode="spectrogram")
        batch = next(iter(train_loader))
        print(f"Spectrogram batch:  input={batch[0].shape}, labels={batch[1].shape}")
    elif dataset == "speechcommands":
        root = sys.argv[2] if len(sys.argv) > 2 else "data"

        train_loader, test_loader = get_speechcommands_dataloaders(root, mode="raw")
        batch = next(iter(train_loader))
        print(f"Raw waveform batch: input={batch[0].shape}, labels={batch[1].shape}")

        train_loader, test_loader = get_speechcommands_dataloaders(root, mode="spectrogram")
        batch = next(iter(train_loader))
        print(f"Spectrogram batch:  input={batch[0].shape}, labels={batch[1].shape}")
    elif dataset == "urbansound8k":
        root = sys.argv[2] if len(sys.argv) > 2 else "data/UrbanSound8K"
        if not os.path.exists(root):
            print(f"UrbanSound8K not found at {root}")
            print("Download from: https://urbansounddataset.weebly.com/urbansound8k.html")
            exit(1)

        train_loader, test_loader = get_urbansound8k_dataloaders(root, test_fold=10, mode="raw")
        batch = next(iter(train_loader))
        print(f"Raw waveform batch: input={batch[0].shape}, labels={batch[1].shape}")

        train_loader, test_loader = get_urbansound8k_dataloaders(root, test_fold=10, mode="spectrogram")
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
