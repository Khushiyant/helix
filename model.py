import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BiMambaBlock(nn.Module):

    def __init__(self, d_model=256, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_forward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_backward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        residual = x
        h = self.norm(x)

        fwd = self.mamba_forward(h)
        bwd = self.mamba_backward(h.flip(dims=[1])).flip(dims=[1])

        combined = torch.cat([fwd, bwd], dim=-1)
        out = self.out_proj(combined)

        return out + residual


class RawWaveformMamba(nn.Module):

    def __init__(
        self,
        num_classes=50,
        d_model=256,
        d_state=32,
        d_conv=4,
        expand=2,
        n_layers=6,
        patch_size=160,
        patch_stride=160,
    ):
        super().__init__()

        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_stride,
        )
        self.input_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            BiMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.transpose(1, 2)
        x = self.input_norm(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


class SpectrogramMamba(nn.Module):

    def __init__(
        self,
        num_classes=50,
        d_model=256,
        d_state=32,
        d_conv=4,
        expand=2,
        n_layers=6,
        n_mels=128,
        patch_h=16,
        patch_w=16,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )
        self.input_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            BiMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)
        x = self.input_norm(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Raw Waveform Model")
    print("=" * 60)
    model_raw = RawWaveformMamba(num_classes=50)
    total_params = sum(p.numel() for p in model_raw.parameters())
    print(f"Total parameters: {total_params:,}")

    dummy_wav = torch.randn(2, 1, 80000)
    out = model_raw(dummy_wav)
    print(f"Input shape:  {dummy_wav.shape}")
    print(f"Output shape: {out.shape}")

    print()
    print("=" * 60)
    print("Testing Spectrogram Model")
    print("=" * 60)
    model_spec = SpectrogramMamba(num_classes=50)
    total_params = sum(p.numel() for p in model_spec.parameters())
    print(f"Total parameters: {total_params:,}")

    dummy_spec = torch.randn(2, 1, 128, 157)
    out = model_spec(dummy_spec)
    print(f"Input shape:  {dummy_spec.shape}")
    print(f"Output shape: {out.shape}")

    print()
    print("Both models working correctly!")
