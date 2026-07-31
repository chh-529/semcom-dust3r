"""
Image-domain JSCC for DUSt3R. (Architecture B)
===============================================

Classic DeepJSCC image transmission (Bourtsoulatze et al. 2019) placed BEFORE
the ViT encoder: the transmitter compresses the raw image with a light CNN,
and the entire DUSt3R network (ViT encoder + cross-view decoder + head) sits
at the receiver.

    Tx:  image (3,H,W) → ImageJSCCEncoder → power-norm → channel
    Rx:  → ImageJSCCDecoder → image_hat → DUSt3R

This contrasts with architecture A (feature_jscc.py), where the SemCom block
compresses ViT *features* between the encoder and the cross-view decoder.

Rate accounting
---------------
The encoder downsamples 4× spatially and emits ``c_out`` channels, i.e.
``c_out·H·W/16`` real symbols per image.  Bandwidth ratio per source dim
(3·H·W pixels) = ``c_out/48``.  To match architecture A's per-image budget of
``k·H·W/256`` symbols (k = feature channel dim), choose  **c_out = k/16**
(e.g. k=128 ↔ c_out=8).

All images are assumed ImgNorm-normalised to [-1, 1] (DUSt3R convention);
the decoder ends in tanh to map back into that range.

Fidelity to the original DeepJSCC (Bourtsoulatze et al. 2019)
-------------------------------------------------------------
The conv/transpose-conv backbone follows the paper: 5 layers, 5×5 kernels,
channels 16→32→32→32→c_out, 4× down/up-sampling, PReLU on every conv layer,
and an average-power constraint on the channel input.  Two intentional
adaptations remain:
  * Output activation is **tanh → [-1, 1]** (not the paper's sigmoid → [0, 1]),
    to match DUSt3R's ImgNorm pixel range.
  * Symbols and the channel are **real-valued** (c_out real channels over a real
    AWGN), whereas the paper uses k complex symbols (2k real units) over a
    complex channel.  Bandwidth here is counted in real symbols.
"""

import math

import torch
import torch.nn as nn

from .channel import AWGNChannel, RayleighChannel


class _ConvPReLU(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=5, stride=stride, padding=2)
        self.act = nn.PReLU(c_out)

    def forward(self, x):
        return self.act(self.conv(x))


class _TConvPReLU(nn.Module):
    def __init__(self, c_in, c_out, stride=1, activate=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            c_in, c_out, kernel_size=5, stride=stride, padding=2,
            output_padding=stride - 1)
        self.act = nn.PReLU(c_out) if activate else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class ImageJSCCEncoder(nn.Module):
    """
    DeepJSCC image encoder (Bourtsoulatze et al. 2019).

    5 conv layers (5×5, channels 16→32→32→32→c_out), stride-2 twice → 4×
    spatial downsampling.  (B, 3, H, W) → (B, c_out, H/4, W/4).  PReLU follows
    every conv layer (including the last, as in the original DeepJSCC; the power
    constraint is enforced afterwards by the power-norm in ImageSemComBlock).
    Fully convolutional, so DUSt3R's variable aspect ratios are handled natively.
    """

    def __init__(self, c_out: int):
        super().__init__()
        self.c_out = c_out
        self.layers = nn.Sequential(
            _ConvPReLU(3, 16, stride=2),
            _ConvPReLU(16, 32, stride=2),
            _ConvPReLU(32, 32),
            _ConvPReLU(32, 32),
            _ConvPReLU(32, c_out),   # PReLU on every conv (DeepJSCC); power-norm follows
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.layers(img)

    @property
    def bandwidth_ratio(self) -> float:
        """Real symbols per source dimension (3·H·W pixels) = c_out/48."""
        return self.c_out / 48.0

    def extra_repr(self):
        return f'c_out={self.c_out}, ratio={self.bandwidth_ratio:.4f}'


class ImageJSCCDecoder(nn.Module):
    """
    DeepJSCC image decoder: strict mirror of the encoder with transposed convs.
    Channel widths 32→32→32→16→3 and strides 1,1,1,2,2 reverse the encoder's
    2,2,1,1,1, so the two stride-2 (upsampling) layers are the last two — exactly
    mirroring the encoder's first two stride-2 (downsampling) layers.
    (B, c_in, H/4, W/4) → (B, 3, H, W) in [-1, 1] (tanh; ImgNorm convention).
    """

    def __init__(self, c_in: int):
        super().__init__()
        self.layers = nn.Sequential(
            _TConvPReLU(c_in, 32),
            _TConvPReLU(32, 32),
            _TConvPReLU(32, 32),
            _TConvPReLU(32, 16, stride=2),
            nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.layers(z))


def _power_normalize_img(z: torch.Tensor):
    """Per-sample unit average power over all symbols.  z: (B, C, h, w)."""
    B = z.shape[0]
    flat = z.reshape(B, -1)
    rms = flat.norm(dim=1).view(B, 1, 1, 1) / math.sqrt(flat.shape[1])
    return z / (rms + 1e-8), rms


class ImageSemComBlock(nn.Module):
    """
    Image-domain SemCom block: acts on raw (normalised) images, mirroring
    SemComBlock's role and interface but for ``DUSt3RSemCom.img_semcom``.

    DeepJSCC mode (``c_out`` int):
        img → CNN encoder → power-norm → channel → CNN decoder → img_hat

    Identity mode (``c_out=None``):  uncoded analog pixel transmission —
        power-norm → channel → inverse-scale.  No learnable parameters.

    Tracks running image PSNR/SSIM (img vs img_hat, computed in [0,1]) so the
    classic SemCom image-quality metrics come for free during evaluation;
    read with ``image_quality_summary()``, clear with ``reset_stats()``.
    """

    def __init__(self, channel: str = 'awgn', snr_db: float = 10.0,
                 c_out: int | None = 8):
        super().__init__()
        self.snr_db = snr_db
        self.c_out = c_out

        channel = channel.lower()
        if channel == 'awgn':
            self.channel = AWGNChannel()
        elif channel == 'rayleigh':
            self.channel = RayleighChannel()
        else:
            raise ValueError(f'Unknown channel {channel!r}')

        if c_out is not None:
            self.encoder = ImageJSCCEncoder(c_out)
            self.decoder = ImageJSCCDecoder(c_out)
        else:
            self.encoder = self.decoder = None

        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._n_imgs = 0

    def forward(self, img: torch.Tensor, snr_db: float | None = None) -> torch.Tensor:
        if snr_db is None:
            snr_db = self.snr_db

        if self.encoder is not None:
            z = self.encoder(img)
            z, _ = _power_normalize_img(z)
            # channel acts elementwise; flatten spatial dims to its (B,N,k) view
            z_hat = self.channel(z.flatten(2), snr_db).view_as(z)
            img_hat = self.decoder(z_hat)
        else:
            z, rms = _power_normalize_img(img)
            z_hat = self.channel(z.flatten(2), snr_db).view_as(z)
            img_hat = z_hat * (rms + 1e-8)

        if not self.training:
            self._track_quality(img, img_hat)
        return img_hat

    # ── image-quality tracking ────────────────────────────────────────────────

    @torch.no_grad()
    def _track_quality(self, img, img_hat):
        from .source_coding import image_psnr_ssim   # avoid import cycle at load
        x01 = (img.float() + 1) / 2
        y01 = (img_hat.float() + 1) / 2
        for b in range(img.shape[0]):
            p, s = image_psnr_ssim(x01[b], y01[b])
            self._psnr_sum += p
            self._ssim_sum += s
            self._n_imgs += 1

    def reset_stats(self):
        self._psnr_sum = self._ssim_sum = 0.0
        self._n_imgs = 0

    def image_quality_summary(self) -> dict:
        n = max(self._n_imgs, 1)
        return {'img_psnr': self._psnr_sum / n, 'img_ssim': self._ssim_sum / n,
                'n_imgs': self._n_imgs}

    @property
    def bandwidth_ratio(self) -> float:
        return 1.0 if self.c_out is None else self.c_out / 48.0

    def rate_summary(self) -> dict:
        return {'c_out': self.c_out, 'bandwidth_ratio': self.bandwidth_ratio}

    def extra_repr(self):
        mode = (f'c_out={self.c_out}, ratio={self.bandwidth_ratio:.4f}'
                if self.c_out is not None else 'identity (uncoded pixels)')
        return f'{mode}, snr_db={self.snr_db}, channel={self.channel.__class__.__name__}'
