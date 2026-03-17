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


def _download_librispeech_hf(root):
    """Download LibriSpeech train-clean-100 via HuggingFace and export WAVs by speaker."""
    index_path = os.path.join(root, "index.csv")
    if os.path.exists(index_path):
        return

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "LibriSpeech auto-download requires the 'datasets' package.\n"
            "  pip install datasets\n"
            "Then re-run."
        )

    import io
    from datasets import Audio

    print("  Downloading LibriSpeech train-clean-100 from HuggingFace...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100")
    # Disable automatic audio decoding (avoids torchcodec/FFmpeg dependency)
    ds = ds.cast_column("audio", Audio(decode=False))

    wav_dir = os.path.join(root, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    rows = []
    print(f"  Exporting {len(ds)} utterances...")
    for i, row in enumerate(ds):
        speaker_id = str(row["speaker_id"])
        audio_bytes = row["audio"]["bytes"]
        samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        duration = len(samples) / sr

        if duration < 0.5:
            continue

        speaker_dir = os.path.join(wav_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        wav_path = os.path.join(speaker_dir, f"{i:06d}.wav")
        sf.write(wav_path, samples, sr)
        rows.append(f"{speaker_id},{wav_path},{duration:.3f}")

        if (i + 1) % 5000 == 0:
            print(f"    {i + 1}/{len(ds)} exported...")

    with open(index_path, "w") as f:
        f.write("speaker_id,path,duration\n")
        f.write("\n".join(rows) + "\n")

    n_speakers = len(set(r.split(",")[0] for r in rows))
    print(f"  LibriSpeech ready: {len(rows)} utterances, {n_speakers} speakers")


def _build_librispeech_speaker_index(root):
    """Build per-speaker segment index from LibriSpeech, auto-downloading if needed."""
    index_path = os.path.join(root, "index.csv")

    if not os.path.exists(index_path):
        # Check for local FLAC files first (manual download)
        subset_dir = os.path.join(root, "train-clean-100")
        if os.path.isdir(subset_dir):
            files, speaker_to_idx = _discover_librispeech_files(root, "train-clean-100")
            speaker_segments = {}
            for path, speaker_id in files:
                try:
                    info = sf.info(path)
                    duration = info.duration
                except Exception:
                    continue
                speaker_segments.setdefault(speaker_id, []).append((path, duration))
            return speaker_segments, speaker_to_idx
        else:
            _download_librispeech_hf(root)

    meta = pd.read_csv(index_path)
    speaker_segments = {}
    for _, row in meta.iterrows():
        speaker_id = str(row["speaker_id"])
        speaker_segments.setdefault(speaker_id, []).append(
            (row["path"], row["duration"])
        )

    speakers = sorted(speaker_segments.keys())
    speaker_to_idx = {s: i for i, s in enumerate(speakers)}
    return speaker_segments, speaker_to_idx


def _split_speakers_by_utterance(speaker_segments, val_ratio=0.2, seed=42):
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


class LibriSpeechScalingRaw(Dataset):
    """Stitches utterances from the same speaker to create long clips
    for scaling experiments (30s, 60s, 300s)."""

    def __init__(self, speaker_segments, speaker_to_idx, target_seconds=30, augment=False):
        self.speaker_to_idx = speaker_to_idx
        self.augment = augment
        self.target_sr = 16000
        self.target_length = int(target_seconds * self.target_sr)

        self.samples = []
        for speaker_id in sorted(speaker_segments.keys()):
            segs = speaker_segments[speaker_id]
            total_dur = sum(dur for _, dur in segs)
            n_samples = max(1, int(total_dur / target_seconds))
            for i in range(n_samples):
                self.samples.append((speaker_id, i, n_samples))

        self.speaker_segments = speaker_segments
        n_speakers = len(speaker_segments)
        print(f"  Loaded {len(self.samples)} samples ({n_speakers} speakers, {target_seconds}s clips)")

    def __len__(self):
        return len(self.samples)

    def _load_wav(self, path):
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            return torch.zeros(1, self.target_sr)
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
            path, _dur = segs[seg_idx]
            chunk = self._load_wav(path)
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


def get_librispeech_scaling_dataloaders(root, target_seconds=30, batch_size=32):
    print(f"\nCreating LibriSpeech scaling dataloaders ({target_seconds}s clips, speaker ID)")
    speaker_segments, speaker_to_idx = _build_librispeech_speaker_index(root)
    train_segs, val_segs = _split_speakers_by_utterance(speaker_segments)

    train_dataset = LibriSpeechScalingRaw(
        train_segs, speaker_to_idx, target_seconds=target_seconds, augment=True,
    )
    test_dataset = LibriSpeechScalingRaw(
        val_segs, speaker_to_idx, target_seconds=target_seconds, augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    return train_loader, test_loader, len(speaker_to_idx)


def _download_voxpopuli_hf(root):
    """Download VoxPopuli English via HuggingFace and export WAVs by speaker."""
    index_train = os.path.join(root, "index_train.csv")
    index_val = os.path.join(root, "index_validation.csv")
    if os.path.exists(index_train) and os.path.exists(index_val):
        return

    try:
        from datasets import load_dataset, Audio
    except ImportError:
        raise RuntimeError(
            "VoxPopuli auto-download requires the 'datasets' package.\n"
            "  pip install datasets\n"
            "Then re-run."
        )

    import io

    wav_dir = os.path.join(root, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    for split_name, index_path in [("train", index_train), ("validation", index_val)]:
        if os.path.exists(index_path):
            continue

        print(f"  Downloading VoxPopuli English ({split_name}) from HuggingFace...")
        ds = load_dataset("facebook/voxpopuli", "en", split=split_name)
        ds = ds.cast_column("audio", Audio(decode=False))

        rows = []
        print(f"  Exporting {len(ds)} utterances ({split_name})...")
        for i, row in enumerate(ds):
            speaker_id = str(row["speaker_id"])
            audio_bytes = row["audio"]["bytes"]
            if audio_bytes is None:
                continue
            try:
                samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            except Exception:
                continue
            duration = len(samples) / sr

            if duration < 1.0:
                continue

            speaker_dir = os.path.join(wav_dir, speaker_id)
            os.makedirs(speaker_dir, exist_ok=True)
            wav_path = os.path.join(speaker_dir, f"{split_name}_{i:06d}.wav")
            sf.write(wav_path, samples, sr)
            rows.append(f"{speaker_id},{wav_path},{duration:.3f}")

            if (i + 1) % 5000 == 0:
                print(f"    {i + 1}/{len(ds)} exported...")

        with open(index_path, "w") as f:
            f.write("speaker_id,path,duration\n")
            f.write("\n".join(rows) + "\n")

        n_speakers = len(set(r.split(",")[0] for r in rows))
        print(f"  VoxPopuli {split_name} ready: {len(rows)} utterances, {n_speakers} speakers")


def _build_voxpopuli_speaker_index(root):
    """Build per-speaker segment index from VoxPopuli, auto-downloading if needed."""
    _download_voxpopuli_hf(root)

    train_segs = {}
    val_segs = {}
    all_speakers = set()

    for split_name, segs_dict in [("train", train_segs), ("validation", val_segs)]:
        index_path = os.path.join(root, f"index_{split_name}.csv")
        meta = pd.read_csv(index_path)
        for _, row in meta.iterrows():
            speaker_id = str(row["speaker_id"])
            segs_dict.setdefault(speaker_id, []).append(
                (row["path"], row["duration"])
            )
            all_speakers.add(speaker_id)

    speaker_to_idx = {s: i for i, s in enumerate(sorted(all_speakers))}
    return train_segs, val_segs, speaker_to_idx


class VoxPopuliScalingRaw(Dataset):
    """Stitches same-speaker VoxPopuli segments to create long clips for scaling experiments.

    VoxPopuli segments are naturally longer than LibriSpeech (tens of seconds vs 5-15s),
    so fewer stitches are needed, producing more natural-sounding clips.
    """

    def __init__(self, speaker_segments, speaker_to_idx, target_seconds=30, augment=False):
        self.speaker_to_idx = speaker_to_idx
        self.augment = augment
        self.target_sr = 16000
        self.target_length = int(target_seconds * self.target_sr)
        self.speaker_segments = speaker_segments

        self.samples = []
        for speaker_id in sorted(speaker_segments.keys()):
            segs = speaker_segments[speaker_id]
            total_dur = sum(dur for _, dur in segs)
            n_samples = max(1, int(total_dur / target_seconds))
            for i in range(n_samples):
                self.samples.append((speaker_id, i, n_samples))

        n_speakers = len(speaker_segments)
        avg_dur = np.mean([d for segs in speaker_segments.values() for _, d in segs])
        avg_stitches = max(1, target_seconds / avg_dur)
        print(f"  {len(self.samples)} samples ({n_speakers} speakers, {target_seconds}s clips, "
              f"~{avg_stitches:.1f} segments/clip avg)")

    def __len__(self):
        return len(self.samples)

    def _load_wav(self, path):
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            print(f"  WARNING: failed to read {path} ({type(e).__name__}): {e}")
            return torch.zeros(1, self.target_sr)
        waveform = torch.from_numpy(data.T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = T.Resample(sr, self.target_sr)(waveform)
        return waveform

    def _augment(self, waveform):
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
            path, _dur = segs[seg_idx]
            chunk = self._load_wav(path)
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


def get_voxpopuli_scaling_dataloaders(root, target_seconds=30, batch_size=32):
    print(f"\nCreating VoxPopuli scaling dataloaders ({target_seconds}s clips, speaker ID)")
    train_segs, val_segs, speaker_to_idx = _build_voxpopuli_speaker_index(root)

    train_dataset = VoxPopuliScalingRaw(
        train_segs, speaker_to_idx, target_seconds=target_seconds, augment=True,
    )
    test_dataset = VoxPopuliScalingRaw(
        val_segs, speaker_to_idx, target_seconds=target_seconds, augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    return train_loader, test_loader, len(speaker_to_idx)


def _download_tedlium_hf(root):
    """Download TEDLIUM 3 via HuggingFace, reconstruct full talks, export as WAVs."""
    index_path = os.path.join(root, "index.csv")
    if os.path.exists(index_path):
        return

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "TEDLIUM auto-download requires the 'datasets' package.\n"
            "  pip install datasets\n"
            "Then re-run."
        )

    wav_dir = os.path.join(root, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    print("  Downloading TEDLIUM 3 from HuggingFace...")
    ds = load_dataset("distil-whisper/tedlium", "release3", split="train", trust_remote_code=True)

    # Pass 1: group segment indices by talk file (no audio decoding)
    print(f"  Scanning {len(ds)} segments into talks...")
    ds_meta = ds.remove_columns(["audio"])
    talks = {}
    for i in range(len(ds_meta)):
        row = ds_meta[i]
        file_id = row.get("file", row["speaker_id"])
        speaker_id = str(row["speaker_id"])
        talks.setdefault(file_id, {"speaker_id": speaker_id, "indices": []})
        talks[file_id]["indices"].append(i)

    # Pass 2: reconstruct each talk by concatenating segments
    print(f"  Reconstructing {len(talks)} talks as WAVs...")
    rows = []
    for talk_idx, (file_id, info) in enumerate(sorted(talks.items())):
        speaker_id = info["speaker_id"]
        parts = []
        sr = 16000
        for seg_idx in info["indices"]:
            audio = ds[seg_idx]["audio"]
            parts.append(np.array(audio["array"], dtype=np.float32))
            sr = audio["sampling_rate"]

        if not parts:
            continue

        full_audio = np.concatenate(parts)
        duration = len(full_audio) / sr

        if duration < 30.0:
            continue

        speaker_dir = os.path.join(wav_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        safe_id = os.path.basename(file_id).replace(".sph", "").replace(" ", "_")
        wav_path = os.path.join(speaker_dir, f"{safe_id}.wav")
        sf.write(wav_path, full_audio, sr)
        rows.append(f"{speaker_id},{wav_path},{duration:.3f}")

        if (talk_idx + 1) % 100 == 0:
            print(f"    {talk_idx + 1}/{len(talks)} talks exported...")

    with open(index_path, "w") as f:
        f.write("speaker_id,path,duration\n")
        f.write("\n".join(rows) + "\n")

    n_speakers = len(set(r.split(",")[0] for r in rows))
    avg_dur = sum(float(r.rsplit(",", 1)[1]) for r in rows) / max(1, len(rows))
    print(f"  TEDLIUM ready: {len(rows)} talks, {n_speakers} speakers, "
          f"avg {avg_dur:.0f}s/talk")


def _build_tedlium_speaker_index(root):
    """Build per-speaker talk index, auto-downloading if needed."""
    _download_tedlium_hf(root)
    meta = pd.read_csv(os.path.join(root, "index.csv"))
    speaker_talks = {}
    for _, row in meta.iterrows():
        speaker_id = str(row["speaker_id"])
        speaker_talks.setdefault(speaker_id, []).append(
            (row["path"], row["duration"])
        )
    speakers = sorted(speaker_talks.keys())
    speaker_to_idx = {s: i for i, s in enumerate(speakers)}
    return speaker_talks, speaker_to_idx


def _split_tedlium_by_talk(speaker_talks, val_ratio=0.2, seed=42):
    """Hold out talks per speaker for validation.
    Speakers with only one talk go to train only."""
    rng = np.random.RandomState(seed)
    train = {}
    val = {}
    for speaker_id in sorted(speaker_talks.keys()):
        talks = list(speaker_talks[speaker_id])
        if len(talks) <= 1:
            train[speaker_id] = talks
        else:
            rng.shuffle(talks)
            n_val = max(1, int(len(talks) * val_ratio))
            val[speaker_id] = talks[:n_val]
            train[speaker_id] = talks[n_val:]
    return train, val


class TedliumScalingRaw(Dataset):
    """Extracts windows from full TEDLIUM talks for scaling experiments.
    Talks are naturally 10-20 minutes — no stitching needed."""

    def __init__(self, speaker_talks, speaker_to_idx, target_seconds=30, augment=False):
        self.speaker_to_idx = speaker_to_idx
        self.augment = augment
        self.target_sr = 16000
        self.target_length = int(target_seconds * self.target_sr)

        self.samples = []
        total = 0
        for speaker_id in sorted(speaker_talks.keys()):
            for path, duration in speaker_talks[speaker_id]:
                total += 1
                if duration >= target_seconds:
                    self.samples.append((speaker_id, path, duration))

        if not self.samples:
            raise ValueError(
                f"No talks >= {target_seconds}s found ({total} total). "
                f"Use a shorter --target_seconds."
            )

        n_speakers = len(set(s for s, _, _ in self.samples))
        print(f"  {len(self.samples)}/{total} talks >= {target_seconds}s ({n_speakers} speakers)")

    def __len__(self):
        return len(self.samples)

    def _load_wav(self, path):
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            print(f"  WARNING: failed to read {path} ({type(e).__name__}): {e}")
            return torch.zeros(1, self.target_length)
        waveform = torch.from_numpy(data.T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = T.Resample(sr, self.target_sr)(waveform)
        return waveform

    def _augment(self, waveform):
        scale = np.random.uniform(0.8, 1.2)
        waveform = waveform * scale
        noise = torch.randn_like(waveform) * 0.005
        waveform = waveform + noise
        return waveform

    def __getitem__(self, idx):
        speaker_id, path, duration = self.samples[idx]
        waveform = self._load_wav(path)
        total = waveform.shape[1]

        if total > self.target_length:
            if self.augment:
                start = np.random.randint(0, total - self.target_length)
            else:
                start = 0
            waveform = waveform[:, start:start + self.target_length]
        elif total < self.target_length:
            pad = self.target_length - total
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        if self.augment:
            waveform = self._augment(waveform)

        return waveform, self.speaker_to_idx[speaker_id]


def get_tedlium_scaling_dataloaders(root, target_seconds=30, batch_size=32):
    print(f"\nCreating TEDLIUM scaling dataloaders ({target_seconds}s clips, speaker ID)")
    speaker_talks, speaker_to_idx = _build_tedlium_speaker_index(root)
    train_talks, val_talks = _split_tedlium_by_talk(speaker_talks)

    train_dataset = TedliumScalingRaw(
        train_talks, speaker_to_idx, target_seconds=target_seconds, augment=True,
    )
    test_dataset = TedliumScalingRaw(
        val_talks, speaker_to_idx, target_seconds=target_seconds, augment=False,
    )

    val_speakers = len(set(s for s, _, _ in test_dataset.samples))
    print(f"  Val covers {val_speakers}/{len(speaker_to_idx)} speakers")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    return train_loader, test_loader, len(speaker_to_idx)


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

    if dataset == "librispeech":
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
