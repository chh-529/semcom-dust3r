"""
Source-coding baselines for DUSt3R feature transmission.
=========================================================

Digital baselines (Shannon separation theorem)
-----------------------------------------------
Error-free delivery is assumed — ideal capacity-achieving channel code at the
given SNR.  This is the standard "separation theorem upper bound" of the
DeepJSCC literature: it is *generous* to digital schemes.  At channel SNR s,
one real symbol carries at most 0.5·log2(1+s) bits, so spending b bits/dim
is equivalent to an analog bandwidth ratio of

    equiv_ratio(b, s) = b / (0.5·log2(1 + 10^(s/10)))

    UniformQuantBlock   uniform scalar quantization, bits/dim rate axis (Arch A).
    ImageCodecBlock     JPEG / WebP image codec before the ViT encoder (Arch B).

Analog baselines  (direct channel simulation)
---------------------------------------------
The selected real symbols are power-normalised and sent over the SAME physical
channel as DeepJSCC.

    AnalogTopKBlock         top-k dims per token + physical channel.
    AnalogTokenPruneBlock   top-k tokens by norm + physical channel.
"""

import io
import math

import numpy as np
import torch
import torch.nn as nn

from .channel import AWGNChannel, RayleighChannel


# ── Shared utilities ──────────────────────────────────────────────────────────

def _make_channel(channel: str):
    c = channel.lower()
    if c == 'awgn':
        return AWGNChannel()
    if c == 'rayleigh':
        return RayleighChannel()
    raise ValueError(f'Unknown channel {channel!r}. Choose "awgn" or "rayleigh".')


def _analog_transmit(z: torch.Tensor, channel, snr_db) -> torch.Tensor:
    """Power-normalise z, pass through channel, inverse-scale.

    Only the symbols actually on the wire should be passed in (e.g. the kept
    top-k entries), so the power constraint is enforced correctly and not
    diluted by structural zeros.  No-op at snr_db=inf.
    """
    if snr_db == float('inf'):
        return z
    B = z.shape[0]
    flat = z.reshape(B, -1)
    rms = (flat.norm(dim=1) / math.sqrt(flat.shape[1])).reshape([B] + [1] * (z.dim() - 1))
    return channel(z / (rms + 1e-8), snr_db) * (rms + 1e-8)


def equiv_analog_ratio(bits_per_dim: float, snr_db: float) -> float:
    """Analog JSCC ratio k/D equivalent to spending ``bits_per_dim`` at ``snr_db``."""
    capacity = 0.5 * math.log2(1.0 + 10 ** (snr_db / 10.0))
    return bits_per_dim / capacity


def image_psnr_ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0):
    """PSNR + SSIM for x, y: (3, H, W) in [0, data_range]."""
    from scipy.ndimage import gaussian_filter
    xn = x.detach().cpu().numpy().astype(np.float64)
    yn = y.detach().cpu().numpy().astype(np.float64)

    mse = ((xn - yn) ** 2).mean()
    psnr = 10.0 * math.log10(data_range ** 2 / max(mse, 1e-12))

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    sigma = 1.5
    ssims = []
    for c in range(xn.shape[0]):
        a, b = xn[c], yn[c]
        mu_a, mu_b = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
        sa  = gaussian_filter(a * a, sigma) - mu_a * mu_a
        sb  = gaussian_filter(b * b, sigma) - mu_b * mu_b
        sab = gaussian_filter(a * b, sigma) - mu_a * mu_b
        smap = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / \
               ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))
        ssims.append(smap.mean())
    return psnr, float(np.mean(ssims))


# ── Digital baselines ─────────────────────────────────────────────────────────

class UniformQuantBlock(nn.Module):
    """
    Uniform scalar quantization of features (quantize → dequantize).

    Quantization range is symmetric per sample, clipped at the
    ``pct`` / (1-``pct``) percentiles to be robust to outliers.

    A histogram of emitted levels is accumulated across forwards;
    ``empirical_entropy()`` gives the bits/dim an ideal entropy coder would
    need — always ≤ the nominal ``bits``.
    """

    def __init__(self, bits: int, pct: float = 0.999):
        super().__init__()
        assert 1 <= bits <= 16
        self.bits = bits
        self.levels = 2 ** bits
        self.pct = pct
        self.snr_db = float('inf')  # interface compat — ignored
        self.register_buffer('hist', torch.zeros(self.levels, dtype=torch.long))

    @torch.no_grad()
    def forward(self, feat: torch.Tensor, snr_db=None) -> torch.Tensor:
        B = feat.shape[0]
        flat = feat.reshape(B, -1).float()
        lo = torch.quantile(flat, 1.0 - self.pct, dim=1, keepdim=True)
        hi = torch.quantile(flat, self.pct, dim=1, keepdim=True)
        step = (hi - lo).clamp(min=1e-8) / (self.levels - 1)
        q = ((flat - lo) / step).round_().clamp_(0, self.levels - 1)
        self.hist += torch.bincount(q.reshape(-1).long(),
                                    minlength=self.levels).to(self.hist.device)
        return (q * step + lo).view_as(feat).to(feat.dtype)

    def reset_stats(self):
        self.hist.zero_()

    def empirical_entropy(self) -> float:
        n = self.hist.sum().item()
        if n == 0:
            return float('nan')
        p = self.hist.double() / n
        p = p[p > 0]
        return float(-(p * p.log2()).sum())

    def rate_summary(self) -> dict:
        ent = self.empirical_entropy()
        return {
            'bits_per_dim': float(self.bits),
            'entropy_per_dim': ent,
            'equiv_ratio_snr0':  equiv_analog_ratio(ent, 0),
            'equiv_ratio_snr10': equiv_analog_ratio(ent, 10),
            'equiv_ratio_snr20': equiv_analog_ratio(ent, 20),
        }

    def extra_repr(self):
        return f'bits={self.bits}, pct={self.pct}'


class ImageCodecBlock(nn.Module):
    """
    Digital baseline: JPEG / WebP image codec before the ViT encoder (Arch B).

    Error-free transmission assumed.  Tracks actual bpp + image PSNR/SSIM so
    codec and JSCC curves are directly comparable via ``rate_summary()``.
    Duck-types ``ImageSemComBlock``: ``forward(img, snr_db=None) → img_hat``
    where img is ImgNorm-normalised to [-1, 1].
    """

    def __init__(self, codec: str = 'jpeg', quality: int = 75):
        super().__init__()
        assert codec in ('jpeg', 'webp')
        self.codec = codec
        self.quality = quality
        self.snr_db = float('inf')  # interface compat — ignored
        self._bits = 0
        self._pixels = 0
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._n_imgs = 0

    @torch.no_grad()
    def forward(self, img: torch.Tensor, snr_db=None) -> torch.Tensor:
        from PIL import Image
        out = torch.empty_like(img)
        for b in range(img.shape[0]):
            x01 = ((img[b].float() + 1) / 2).clamp(0, 1)
            arr = (x01 * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format=self.codec.upper(),
                                      quality=self.quality)
            self._bits += 8 * buf.getbuffer().nbytes
            self._pixels += arr.shape[0] * arr.shape[1]
            dec = Image.open(buf)
            y01 = torch.from_numpy(np.array(dec)).permute(2, 0, 1).float() / 255
            y01 = y01.to(img.device)
            p, s = image_psnr_ssim(x01, y01)
            self._psnr_sum += p
            self._ssim_sum += s
            self._n_imgs += 1
            out[b] = (y01 * 2 - 1).to(img.dtype)
        return out

    def reset_stats(self):
        self._bits = self._pixels = 0
        self._psnr_sum = self._ssim_sum = 0.0
        self._n_imgs = 0

    def image_quality_summary(self) -> dict:
        n = max(self._n_imgs, 1)
        return {'img_psnr': self._psnr_sum / n, 'img_ssim': self._ssim_sum / n,
                'n_imgs': self._n_imgs}

    @property
    def bits_per_pixel(self) -> float:
        return self._bits / max(self._pixels, 1)

    def rate_summary(self) -> dict:
        bpp = self.bits_per_pixel
        bits_per_dim = bpp / 3.0
        return {
            'codec': self.codec,
            'quality': self.quality,
            'bits_per_pixel': bpp,
            'bits_per_dim': bits_per_dim,
            'equiv_ratio_snr0':  equiv_analog_ratio(bits_per_dim, 0),
            'equiv_ratio_snr10': equiv_analog_ratio(bits_per_dim, 10),
            'equiv_ratio_snr20': equiv_analog_ratio(bits_per_dim, 20),
        }

    def extra_repr(self):
        return f'codec={self.codec}, quality={self.quality}'


class BudgetJPEGBlock(nn.Module):
    """
    Digital baseline: JPEG with SNR-derived bit budget (cliff effect).

    For a given channel SNR, the maximum bits a digital system can reliably
    deliver under the same bandwidth as Arch B is:

        bpp = bandwidth_ratio × 0.5·log2(1 + SNR_lin) × 3   (real AWGN, RGB)

    The highest JPEG quality whose compressed size fits within this budget is
    found by binary search per image.  If even quality=1 exceeds the budget,
    the image is replaced by a uniform gray frame (outage — the cliff effect).

    ``bandwidth_ratio`` = c_out / 48  (8/48 = 1/6 for Arch B c_out=8).

    Duck-types ``ImageSemComBlock``:  forward(img, snr_db=None) → img_hat.
    """

    def __init__(self, bandwidth_ratio: float = 8 / 48.0, snr_db: float = 10.0):
        super().__init__()
        self.bandwidth_ratio = bandwidth_ratio
        self.snr_db = snr_db
        self._bits = 0
        self._pixels = 0
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._n_imgs = 0
        self._n_outage = 0

    def _bpp_budget(self, snr_db: float) -> float:
        if snr_db == float('inf'):
            return float('inf')
        snr_lin = 10 ** (snr_db / 10.0)
        return self.bandwidth_ratio * 0.5 * math.log2(1.0 + snr_lin) * 3.0

    @torch.no_grad()
    def forward(self, img: torch.Tensor, snr_db=None) -> torch.Tensor:
        from PIL import Image
        if snr_db is None:
            snr_db = self.snr_db
        bpp_limit = self._bpp_budget(snr_db)
        out = torch.empty_like(img)
        for b in range(img.shape[0]):
            x01 = ((img[b].float() + 1) / 2).clamp(0, 1)
            arr = (x01 * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            h, w = arr.shape[:2]
            budget_bytes = bpp_limit * h * w / 8.0

            # binary search: highest quality whose file size ≤ budget
            lo, hi, best_q = 1, 95, None
            while lo <= hi:
                mid = (lo + hi) // 2
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format='JPEG', quality=mid)
                if buf.tell() <= budget_bytes:
                    best_q = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            if best_q is None:
                # outage: gray frame
                dec_arr = np.full_like(arr, 128)
                self._n_outage += 1
                y01 = torch.full_like(x01, 0.5)
            else:
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format='JPEG', quality=best_q)
                self._bits += 8 * buf.tell()
                buf.seek(0)
                dec_arr = np.array(Image.open(buf))
                y01 = torch.from_numpy(dec_arr).permute(2, 0, 1).float() / 255
                y01 = y01.to(img.device)

            self._pixels += h * w
            p, s = image_psnr_ssim(x01, y01)
            self._psnr_sum += p
            self._ssim_sum += s
            self._n_imgs += 1
            out[b] = (y01 * 2 - 1).to(img.dtype)
        return out

    def reset_stats(self):
        self._bits = self._pixels = 0
        self._psnr_sum = self._ssim_sum = 0.0
        self._n_imgs = self._n_outage = 0

    def image_quality_summary(self) -> dict:
        n = max(self._n_imgs, 1)
        return {'img_psnr': self._psnr_sum / n, 'img_ssim': self._ssim_sum / n,
                'n_imgs': self._n_imgs, 'n_outage': self._n_outage}

    @property
    def bits_per_pixel(self) -> float:
        return self._bits / max(self._pixels, 1)

    def rate_summary(self) -> dict:
        bpp = self.bits_per_pixel
        budget = self._bpp_budget(self.snr_db)
        return {
            'bandwidth_ratio': self.bandwidth_ratio,
            'snr_db': self.snr_db,
            'bpp_budget': budget,
            'bits_per_pixel': bpp,
            'n_outage': self._n_outage,
        }

    def extra_repr(self):
        return (f'bandwidth_ratio={self.bandwidth_ratio:.4f}, snr_db={self.snr_db}, '
                f'bpp_budget={self._bpp_budget(self.snr_db):.3f}')


# ── Analog baselines ──────────────────────────────────────────────────────────

class AnalogTopKBlock(nn.Module):
    """
    Analog baseline: per-token top-k selection + physical channel.

    The k largest-|value| dims per token are power-normalised and sent over
    the physical channel at the specified SNR; all other dims are zeroed.
    Bandwidth ratio = k/D, same as DeepJSCC at compression ratio k/D.

    The index set (which dims to keep) is treated as error-free side
    information — the same generous assumption the digital baseline makes for
    its bits.  This isolates the value of the JSCC learned encoder/decoder
    relative to a heuristic selection rule.
    """

    def __init__(self, keep_frac: float, feat_dim: int = 1024,
                 channel: str = 'awgn', snr_db: float = float('inf')):
        super().__init__()
        assert 0.0 < keep_frac <= 1.0
        self.keep_frac = keep_frac
        self.feat_dim = feat_dim
        self.k = max(1, round(keep_frac * feat_dim))
        self.snr_db = snr_db
        self.channel = _make_channel(channel)

    @torch.no_grad()
    def forward(self, feat: torch.Tensor, snr_db=None) -> torch.Tensor:
        if snr_db is None:
            snr_db = self.snr_db
        idx = feat.abs().topk(self.k, dim=-1).indices
        vals = _analog_transmit(feat.gather(-1, idx), self.channel, snr_db)
        out = torch.zeros_like(feat)
        out.scatter_(-1, idx, vals)
        return out

    def rate_summary(self) -> dict:
        return {
            'keep_frac': self.keep_frac,
            'kept_dims': self.k,
            'bandwidth_ratio': self.k / self.feat_dim,
        }

    def extra_repr(self):
        return (f'keep_frac={self.keep_frac} (k={self.k}/{self.feat_dim}), '
                f'analog/{self.channel.__class__.__name__}, snr_db={self.snr_db}')


class AnalogTokenPruneBlock(nn.Module):
    """
    Analog baseline: token-level pruning + physical channel.

    Keeps ``keep_frac`` of the N tokens (by feature L2 norm); the kept tokens
    (full D-dim vectors) are power-normalised and sent over the physical
    channel.  Bandwidth ratio = keep_frac (same as DeepJSCC at that ratio).

    The kept-token index set is treated as error-free side information.
    """

    def __init__(self, keep_frac: float, importance: str = 'norm',
                 channel: str = 'awgn', snr_db: float = float('inf')):
        super().__init__()
        assert 0.0 < keep_frac <= 1.0
        assert importance in ('norm', 'random')
        self.keep_frac = keep_frac
        self.importance = importance
        self.snr_db = snr_db
        self.channel = _make_channel(channel)

    @torch.no_grad()
    def forward(self, feat: torch.Tensor, snr_db=None) -> torch.Tensor:
        if snr_db is None:
            snr_db = self.snr_db
        B, N, D = feat.shape
        k = max(1, round(self.keep_frac * N))
        scores = feat.float().norm(dim=-1) if self.importance == 'norm' \
            else torch.rand(B, N, device=feat.device)
        idx = scores.topk(k, dim=1).indices
        idx_e = idx.unsqueeze(-1).expand(-1, -1, D)
        kept = _analog_transmit(feat.gather(1, idx_e), self.channel, snr_db)
        out = torch.zeros_like(feat)
        out.scatter_(1, idx_e, kept)
        return out

    def rate_summary(self) -> dict:
        return {
            'keep_frac': self.keep_frac,
            'importance': self.importance,
            'bandwidth_ratio': self.keep_frac,
        }

    def extra_repr(self):
        return (f'keep_frac={self.keep_frac}, importance={self.importance}, '
                f'analog/{self.channel.__class__.__name__}, snr_db={self.snr_db}')
