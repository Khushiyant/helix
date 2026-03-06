import torch
import torch.nn as nn
from mamba_ssm import Mamba


class SelfAttentionBlock(nn.Module):

    def __init__(self, d_model=256, num_heads=4, ffn_dim=None):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = d_model * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x):
        h = self.norm1(x)
        out, _ = self.attn(h, h, h)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x


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
        attention_at=(),
        num_heads=4,
    ):
        super().__init__()

        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_stride,
        )
        self.input_norm = nn.LayerNorm(d_model)

        # Match attention block params to BiMamba block params
        ref = BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        mamba_params = sum(p.numel() for p in ref.parameters())
        attn_fixed = 4 * d_model**2 + 4 * d_model + 4 * d_model  # MHA + 2 norms
        ffn_dim = (mamba_params - attn_fixed - d_model) // (2 * d_model + 1)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i in attention_at:
                self.layers.append(SelfAttentionBlock(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim))
            else:
                self.layers.append(BiMambaBlock(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                ))

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, n_pool_tokens=None):
        x = self.patch_embed(x)
        x = x.transpose(1, 2)
        x = self.input_norm(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        x = x[:, :n_pool_tokens, :].mean(dim=1) if n_pool_tokens is not None else x.mean(dim=1)
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
        attention_at=(),
        num_heads=4,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )
        self.input_norm = nn.LayerNorm(d_model)

        ref = BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        mamba_params = sum(p.numel() for p in ref.parameters())
        attn_fixed = 4 * d_model**2 + 4 * d_model + 4 * d_model
        ffn_dim = (mamba_params - attn_fixed - d_model) // (2 * d_model + 1)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i in attention_at:
                self.layers.append(SelfAttentionBlock(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim))
            else:
                self.layers.append(BiMambaBlock(
                    d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                ))

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, n_pool_tokens=None):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)
        x = self.input_norm(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        x = x[:, :n_pool_tokens, :].mean(dim=1) if n_pool_tokens is not None else x.mean(dim=1)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    configs = [
        ("Raw Waveform (Pure Mamba)", lambda: RawWaveformMamba(num_classes=50), torch.randn(2, 1, 80000)),
        ("Raw Waveform (HELIX)", lambda: RawWaveformMamba(num_classes=50, attention_at=(3,)), torch.randn(2, 1, 80000)),
        ("Raw Waveform (Pure Attention)", lambda: RawWaveformMamba(num_classes=50, attention_at=tuple(range(6))), torch.randn(2, 1, 80000)),
        ("Spectrogram (Pure Mamba)", lambda: SpectrogramMamba(num_classes=50), torch.randn(2, 1, 128, 157)),
        ("Spectrogram (HELIX)", lambda: SpectrogramMamba(num_classes=50, attention_at=(3,)), torch.randn(2, 1, 128, 157)),
    ]

    for name, make_model, dummy in configs:
        print("=" * 60)
        print(f"Testing {name}")
        print("=" * 60)
        model = make_model()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")
        out = model(dummy)
        print(f"  Input:  {dummy.shape}")
        print(f"  Output: {out.shape}")
        print(f"  Layers: {[type(l).__name__ for l in model.layers]}")
        print()

    print("All models working correctly!")
