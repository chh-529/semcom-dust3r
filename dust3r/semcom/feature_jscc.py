"""
Feature-domain JSCC for DUSt3R. (Architecture A)
=================================================

learnable JSCC encoder/decoder compress
ViT encoder features between the encoder and cross-view decoder, then the
compressed symbols pass through the physical channel.

    Tx:  feat (B, N, D) → JSCCEncoder → power-norm → channel
    Rx:  → JSCCDecoder → feat_hat (B, N, D) → DUSt3R decoder


JSCCEncoder / JSCCDecoder
--------------------------

    Two-layer MLP with GELU non-linearity and LayerNorm, following the
    DeepJSCC architecture (Bourtsoulatze et al. 2019) adapted for semantic
    features (Xie et al. DEEPSC 2021).

    Encoder: LayerNorm(D) → Linear(D,H) → GELU → Linear(H,k)
    Decoder: Linear(k,H) → GELU → Linear(H,D) → LayerNorm(D)

The compression ratio is k / D.  Power normalisation is applied inside
SemComBlock so the channel power constraint is enforced in a unified place.

SemComBlock
-----------
Wraps a JSCC encoder/decoder pair with the physical channel.  Exposes
``forward(feat, snr_db=None) -> feat_hat`` so it duck-types with the
source-coding baselines in source_coding.py.

Identity mode (``channel_dim=None``): no learnable params; power-normalise
the raw features and pass them through the channel.  Useful as a
noise-injection baseline at full bandwidth.
"""

import math

import torch
import torch.nn as nn

from .channel import AWGNChannel, RayleighChannel


class JSCCEncoder(nn.Module):
    """
    Non-linear JSCC Encoder: feat (B, N, D) → z (B, N, k).

    Architecture: LayerNorm(D) → Linear(D, H) → GELU → Linear(H, k)

    Args:
        feat_dim    : Input feature dimension D (e.g. 1024 for ViT-Large).
        channel_dim : Output channel symbol dimension k.
        hidden_dim  : Hidden layer width H.  Defaults to ``feat_dim``.
    """

    def __init__(self, feat_dim: int, channel_dim: int,
                 hidden_dim: int | None = None):
        super().__init__()
        self.feat_dim    = feat_dim
        self.channel_dim = channel_dim
        self.hidden_dim  = hidden_dim if hidden_dim is not None else feat_dim

        self.encoder = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, channel_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.encoder(feat)

    @property
    def compression_ratio(self) -> float:
        return self.channel_dim / self.feat_dim

    def extra_repr(self) -> str:
        return (f'feat_dim={self.feat_dim}, hidden_dim={self.hidden_dim}, '
                f'channel_dim={self.channel_dim}, ratio={self.compression_ratio:.4f}')


class JSCCDecoder(nn.Module):
    """
    Non-linear JSCC Decoder: z_hat (B, N, k) → feat_hat (B, N, D).

    Architecture: Linear(k, H) → GELU → Linear(H, D) → LayerNorm(D)

    Args:
        channel_dim : Input channel symbol dimension k.
        feat_dim    : Output feature dimension D.
        hidden_dim  : Hidden layer width H.  Defaults to ``feat_dim``.
    """

    def __init__(self, channel_dim: int, feat_dim: int,
                 hidden_dim: int | None = None):
        super().__init__()
        self.channel_dim = channel_dim
        self.feat_dim    = feat_dim
        self.hidden_dim  = hidden_dim if hidden_dim is not None else feat_dim

        self.decoder = nn.Sequential(
            nn.Linear(channel_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_hat)

    def extra_repr(self) -> str:
        return (f'channel_dim={self.channel_dim}, hidden_dim={self.hidden_dim}, '
                f'feat_dim={self.feat_dim}')


# ── SemComBlock: full Arch A pipeline ────────────────────────────────────────

def _power_normalize(z: torch.Tensor):
    """Per-sample unit average power.  z: (B, N, D) → (z_norm, rms)."""
    B = z.shape[0]
    flat = z.view(B, -1)
    rms = flat.norm(dim=1).unsqueeze(-1).unsqueeze(-1) / math.sqrt(flat.shape[1])
    return z / (rms + 1e-8), rms


class SemComBlock(nn.Module):
    """
    Semantic Communication block for DUSt3R (architecture A).

    Sits between the ViT encoder and cross-view decoder.

    DeepJSCC mode (``channel_dim=k``):
        feat → JSCCEncoder(D→k) → power-norm → channel → JSCCDecoder(k→D) → feat_hat

    Identity mode (``channel_dim=None``):
        feat → power-norm → channel → inverse-scale → feat_hat
        No learnable parameters; useful as a noise-injection baseline.

    Args:
        channel     : ``'awgn'`` or ``'rayleigh'``.
        snr_db      : Default SNR in dB.  ``float('inf')`` = noiseless.
        feat_dim    : Input feature dimension D (1024 for ViT-Large).
        channel_dim : Channel symbol dimension k.  ``None`` → identity mode.
        hidden_dim  : MLP hidden width H.  Defaults to ``feat_dim``.
    """

    def __init__(
        self,
        channel: str = 'awgn',
        snr_db: float = 10.0,
        feat_dim: int = 1024,
        channel_dim: int | None = None,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.snr_db      = snr_db
        self.feat_dim    = feat_dim
        self.channel_dim = channel_dim

        channel_lower = channel.lower()
        if channel_lower == 'awgn':
            self.channel = AWGNChannel()
        elif channel_lower == 'rayleigh':
            self.channel = RayleighChannel()
        else:
            raise ValueError(f'Unknown channel {channel!r}. Choose "awgn" or "rayleigh".')

        if channel_dim is not None:
            self.jscc_encoder = JSCCEncoder(feat_dim, channel_dim, hidden_dim)
            self.jscc_decoder = JSCCDecoder(channel_dim, feat_dim, hidden_dim)
        else:
            self.jscc_encoder = None
            self.jscc_decoder = None

    def forward(self, feat: torch.Tensor, snr_db: float | None = None) -> torch.Tensor:
        """
        Args:
            feat   : (B, N, D)
            snr_db : SNR override in dB.
        Returns:
            feat_hat : (B, N, D)
        """
        if snr_db is None:
            snr_db = self.snr_db

        if self.jscc_encoder is not None:
            z = self.jscc_encoder(feat)
            z_norm, _ = _power_normalize(z)
            z_hat = self.channel(z_norm, snr_db)
            return self.jscc_decoder(z_hat)
        else:
            z, rms = _power_normalize(feat)
            z_hat = self.channel(z, snr_db)
            return z_hat * (rms + 1e-8)

    @property
    def compression_ratio(self) -> float:
        if self.channel_dim is None:
            return 1.0
        return self.channel_dim / self.feat_dim

    def extra_repr(self) -> str:
        mode = (f'feat_dim={self.feat_dim}, channel_dim={self.channel_dim}, '
                f'ratio={self.compression_ratio:.4f}'
                if self.channel_dim is not None else 'identity (no compression)')
        return f'{mode}, snr_db={self.snr_db}, channel={self.channel.__class__.__name__}'
