# --------------------------------------------------------
# DUSt3R + Semantic Communication
# --------------------------------------------------------
import os

import numpy as np
import torch
import tqdm

from dust3r.inference import check_if_same_size
from dust3r.model import AsymmetricCroCo3DStereo, load_model
from dust3r.semcom import SemComBlock, ImageSemComBlock
from dust3r.utils.device import collate_with_cat, to_cpu
from dust3r.utils.misc import freeze_all_params


class DUSt3RSemCom(AsymmetricCroCo3DStereo):
    """
    DUSt3R extended with a Semantic Communication channel.

    Architecture A — feature-domain JSCC  (``feat_semcom`` is set)
    ───────────────────────────────────────────────────────────────
        Tx │ Image → [ViT Encoder] → feat (B,N,1024) → [JSCC Enc · MLP  1024→k] → power-norm → ┐
                                                                                               │ AWGN / Rayleigh
        Rx │ ┌─────────────────────────────────────────────────────────────────────────────────┘
           │ └→ [JSCC Dec · MLP  k→1024] → feat_hat → [Cross-view Decoder] → [Head] → pred

    Architecture B — image-domain JSCC  (``img_semcom`` is set)
    ────────────────────────────────────────────────────────────
        Tx │ Image → [SemCom Enc · CNN] → power-norm → ┐
                                                       │ AWGN / Rayleigh
        Rx │ ┌─────────────────────────────────────────┘
           │ └→ [SemCom Dec · CNN] → img_hat → [ViT Encoder] → [Cross-view Decoder] → [Head] → pred

    Baseline: setting both to ``None`` gives a clean DUSt3R with no channel effects.

    Args:
        feat_semcom : SemComBlock or None — Arch A feature-domain channel block,
                      inserted between ViT encoder and cross-view decoder.
        img_semcom  : ImageSemComBlock or None — Arch B image-domain channel
                      block, applied to raw images before the ViT encoder.
        **kwargs    : Forwarded to AsymmetricCroCo3DStereo.__init__.
    """

    def __init__(self, feat_semcom=None, img_semcom=None, **kwargs):
        super().__init__(**kwargs)
        self.feat_semcom = feat_semcom
        self.img_semcom = img_semcom
        self.encoder_frozen = False

    def forward(self, view1, view2):
        # Arch B: image-domain channel before the ViT encoder.
        if self.img_semcom is not None:
            view1 = dict(view1, img=self.img_semcom(view1['img']))
            view2 = dict(view2, img=self.img_semcom(view2['img']))

        # ViT encoder
        with torch.set_grad_enabled(not self.encoder_frozen):
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # Arch A: feature-domain channel between encoder and decoder.
        if self.feat_semcom is not None:
            feat1 = self.feat_semcom(feat1)
            feat2 = self.feat_semcom(feat2)

        # Cross-view decoder + prediction head (receiver side for both archs).
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.amp.autocast('cuda', enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')
        return res1, res2

    @torch.no_grad()
    def transmit(self, views, device):
        """
        views  : list of image dicts (from load_images), each with a unique 'idx'.
        Returns: {idx: (feat_hat, pos, true_shape)} — receiver-side feature cache.
        """
        cache = {} # cache the receiver-side features
        for view in views:
            img = view['img'].to(device, non_blocking=True)
            true_shape = view.get('true_shape', torch.tensor(img.shape[-2:])[None].repeat(len(img), 1))
            if isinstance(true_shape, np.ndarray):
                true_shape = torch.from_numpy(true_shape)
            if self.img_semcom is not None:
                img = self.img_semcom(img)
            feat, pos, _ = self._encode_image(img, true_shape)
            if self.feat_semcom is not None:
                feat = self.feat_semcom(feat)
            cache[view['idx']] = (feat, pos, true_shape)
        return cache

    @torch.no_grad()
    def decode_pair(self, feat1, pos1, shape1, feat2, pos2, shape2):
        """ Receiver side: cross-view decoder + head """
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        with torch.amp.autocast('cuda', enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
        res2['pts3d_in_other_view'] = res2.pop('pts3d')
        return res1, res2


@torch.no_grad()
def run_inference(pairs, model, device, batch_size=8, verbose=True):
    """
    pairs : list of (view1, view2) image-dict tuples (e.g. from make_pairs).
    model : DUSt3RSemCom.
    """
    assert isinstance(model, DUSt3RSemCom), 'This inference requires a DUSt3RSemCom model'

    # 1. Transmitter: one channel use per unique view.
    unique_views = {}
    for v1, v2 in pairs:
        for v in (v1, v2):
            unique_views.setdefault(v['idx'], v)
    if verbose:
        print(f'>> Inference: {len(unique_views)} views, {len(pairs)} pairs')
    cache = model.transmit(list(unique_views.values()), device)

    # 2. Receiver: decode every pair from the cached received features.
    multiple_shapes = not check_if_same_size(pairs)
    if multiple_shapes:  # force bs=1
        batch_size = 1

    result = []
    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        chunk = pairs[i:i + batch_size]
        feat1, pos1, shape1 = (torch.cat(t) for t in
                               zip(*(cache[v1['idx']] for v1, _ in chunk)))
        feat2, pos2, shape2 = (torch.cat(t) for t in
                               zip(*(cache[v2['idx']] for _, v2 in chunk)))
        pred1, pred2 = model.decode_pair(feat1, pos1, shape1, feat2, pos2, shape2)

        view1, view2 = collate_with_cat(chunk)
        res = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=None)
        result.append(to_cpu(res))

    return collate_with_cat(result, lists=multiple_shapes)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _transplant(base, feat_semcom, img_semcom=None) -> DUSt3RSemCom:
    """Transplant backbone weights + semcom block(s) into a DUSt3RSemCom instance."""
    model = DUSt3RSemCom.__new__(DUSt3RSemCom)
    model.__dict__.update(base.__dict__)
    model.feat_semcom = feat_semcom
    model.img_semcom = img_semcom
    model.__class__ = DUSt3RSemCom
    return model


# ── Public API ────────────────────────────────────────────────────────────────

def load_semcom_model(
    dust3r_path: str,
    device: str,
    snr_db: float = float('inf'),
    channel: str = 'awgn',
    jscc_path: str | None = None,
    verbose: bool = True,
) -> DUSt3RSemCom:
    """
    Load a DUSt3R + SemCom model for inference.

    The architecture (A or B) is determined automatically from the checkpoint's
    saved config (``config['domain'] == 'image'`` → Arch B; otherwise Arch A).

    Args:
        dust3r_path : Path to the original DUSt3R .pth checkpoint.
        device      : ``'cuda'`` or ``'cpu'``.
        snr_db      : Inference SNR in dB.  Overrides the value stored in the
                      checkpoint (allows evaluating a model at a different SNR).
        channel     : ``'awgn'`` or ``'rayleigh'`` — only used in noise-only mode
                      (when ``jscc_path`` is ``None`` and ``snr_db`` is finite).
        jscc_path   : Path to a JSCC-only or full-model checkpoint, or ``None``
                      for the clean / noise-only baselines.
        verbose     : Print loading messages.

    Returns:
        model : DUSt3RSemCom in eval mode on *device*.
    """
    base = load_model(dust3r_path, device='cpu', verbose=verbose)

    if jscc_path is None:
        # ── No checkpoint: noise-only or clean baseline ────────────────────
        if snr_db == float('inf'):
            semcom = None
            if verbose:
                print('[SemCom] SNR=inf → clean DUSt3R baseline (no channel block)')
        else:
            semcom = SemComBlock(channel=channel, snr_db=snr_db)
            if verbose:
                print(f'[SemCom] Noise-only | channel={channel}, SNR={snr_db} dB')
        model = _transplant(base, semcom)

    else:
        # ── Load from checkpoint ──────────────────────────────────────────
        ckpt = torch.load(jscc_path, map_location='cpu', weights_only=False)
        cfg  = ckpt['config']

        if cfg.get('domain') == 'image':
            # Architecture B: image-domain JSCC before the (receiver-side) DUSt3R.
            img_semcom = ImageSemComBlock(channel=cfg['channel'], snr_db=snr_db,
                                          c_out=cfg['c_out'])
            model = _transplant(base, None, img_semcom)
            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'], strict=True)
                model.img_semcom.snr_db = snr_db
            else:
                img_semcom.load_state_dict(ckpt['jscc_state_dict'])
            if verbose:
                print(f'[SemCom] Loaded IMAGE-domain JSCC from {jscc_path}\n'
                      f'  channel={cfg["channel"]}, c_out={cfg["c_out"]} '
                      f'(ratio={img_semcom.bandwidth_ratio:.4f}), '
                      f'inference SNR={snr_db} dB')
            return model.to(device).eval()

        semcom = SemComBlock(
            channel=cfg['channel'],
            snr_db=snr_db,
            feat_dim=cfg['feat_dim'],
            channel_dim=cfg['channel_dim'],
            hidden_dim=cfg.get('hidden_dim'),
        )

        if verbose:
            if cfg['channel_dim'] is None:
                print(
                    f'[SemCom] Loaded from {jscc_path}\n'
                    f'  channel={cfg["channel"]}, identity (no JSCC compression), '
                    f'inference SNR={snr_db} dB'
                )
            else:
                ratio = cfg['channel_dim'] / cfg['feat_dim']
                print(
                    f'[SemCom] Loaded from {jscc_path}\n'
                    f'  channel={cfg["channel"]}, channel_dim={cfg["channel_dim"]}'
                    f' (ratio={ratio:.3f}), inference SNR={snr_db} dB'
                )

        model = _transplant(base, semcom)

        if 'model' in ckpt:
            # Full fine-tuned checkpoint (backbone + SemCom jointly trained).
            # Back-compat: checkpoints saved before the feat_semcom rename use the
            # old "semcom." prefix; remap so they still load strict.
            state = {
                (f'feat_semcom.{k[len("semcom."):]}' if k.startswith('semcom.') else k): v
                for k, v in ckpt['model'].items()
            }
            model.load_state_dict(state, strict=True)
            model.feat_semcom.snr_db = snr_db
        else:
            # JSCC-only checkpoint (frozen backbone training)
            semcom.load_state_dict(ckpt['jscc_state_dict'])

    return model.to(device).eval()


def build_semcom_model(
    dust3r_path: str,
    device: str = 'cpu',
    freeze: str = 'none',
    snr_db: float = 10.0,
    channel: str = 'awgn',
    channel_dim: int = 512,
    hidden_dim: int | None = None,
    feat_dim: int = 1024,
    jscc_path: str | None = None,
    domain: str = 'feature',
    img_c_out: int | None = 8,
    verbose: bool = True,
) -> tuple:
    """
    Build a DUSt3RSemCom model ready for training.

    Args:
        dust3r_path : Path to the original DUSt3R .pth checkpoint.
        device      : 'cuda' or 'cpu'.
        domain      : 'feature' or 'image'
        freeze      : Backbone freeze strategy:
                        'encoder' — freeze ViT patch_embed + enc_blocks;
                                        train JSCC + decoder + heads.
                        'none'    — all parameters trainable (full joint).
        snr_db      : Default training SNR in dB (typically sampled per batch).
        channel     : 'awgn' or 'rayleigh'.
        img_c_out   : [Arch B only] CNN output channels; controls bandwidth ratio
                      (ratio = c_out / 48).  Budget-match to Arch A via c_out = k/16.
        channel_dim : [Arch A only] JSCC channel symbol dimension k
                      (ratio = k / feat_dim).
        hidden_dim  : [Arch A only] JSCC MLP hidden width (defaults to feat_dim).
        feat_dim    : [Arch A only] ViT encoder feature dimension D.
        jscc_path   : Optional checkpoint for warm-start.  Accepts both JSCC-only
                      (``jscc_state_dict`` key) and full-model (``model`` key).
        verbose     : Print loading messages.

    Returns:
        (model, jscc_config)
        model       : DUSt3RSemCom in train mode on *device*.
        jscc_config : dict of JSCC hyper-parameters saved alongside the checkpoint.
    """
    base = load_model(dust3r_path, device='cpu', verbose=verbose)

    if domain == 'image':
        # Architecture B: CNN JSCC on raw images, whole DUSt3R at receiver.
        img_semcom = ImageSemComBlock(channel=channel, snr_db=snr_db, c_out=img_c_out)
        if jscc_path is not None and os.path.exists(jscc_path):
            ckpt = torch.load(jscc_path, map_location='cpu', weights_only=False)
            img_semcom.load_state_dict(ckpt['jscc_state_dict'])
            if verbose:
                print(f'[SemCom] image-JSCC warm-start from {jscc_path}')
        jscc_config = {'domain': 'image', 'c_out': img_c_out, 'channel': channel}
        model = _transplant(base, None, img_semcom)
        if freeze == 'encoder':
            freeze_all_params([model.mask_token, model.patch_embed, model.enc_blocks])
            model.encoder_frozen = True
        if verbose:
            ratio = img_semcom.bandwidth_ratio
            print(f'[SemCom] IMAGE-domain JSCC: c_out={img_c_out} (ratio={ratio:.4f}), '
                  f'channel={channel}')
        return model.to(device).train(), jscc_config

    if jscc_path is not None and os.path.exists(jscc_path):
        ckpt = torch.load(jscc_path, map_location='cpu', weights_only=False)
        if 'jscc_state_dict' not in ckpt:
            raise ValueError(
                f'Cannot read JSCC weights from {jscc_path!r}: '
                '"jscc_state_dict" key not found.'
            )
        cfg = ckpt['config']
        semcom = SemComBlock(
            channel=cfg['channel'],
            snr_db=snr_db,
            feat_dim=cfg['feat_dim'],
            channel_dim=cfg['channel_dim'],
            hidden_dim=cfg.get('hidden_dim'),
        )
        semcom.load_state_dict(ckpt['jscc_state_dict'])
        jscc_config = {
            'feat_dim':    cfg['feat_dim'],
            'channel_dim': cfg['channel_dim'],
            'hidden_dim':  cfg.get('hidden_dim'),
            'channel':     cfg['channel'],
        }
        if verbose:
            print(f'[SemCom] JSCC warm-start from {jscc_path}')
    else:
        semcom = SemComBlock(
            channel=channel,
            snr_db=snr_db,
            feat_dim=feat_dim,
            channel_dim=channel_dim,
            hidden_dim=hidden_dim,
        )
        jscc_config = {
            'feat_dim':    feat_dim,
            'channel_dim': channel_dim,
            'hidden_dim':  hidden_dim,
            'channel':     channel,
        }
        if verbose:
            if channel_dim is None:
                print(f'[SemCom] Identity channel (no JSCC compression), channel={channel}')
            else:
                print(f'[SemCom] Fresh JSCC: k={channel_dim}, channel={channel}')

    model = _transplant(base, semcom)

    if freeze == 'encoder':
        freeze_all_params([model.mask_token, model.patch_embed, model.enc_blocks])
        model.encoder_frozen = True
        if verbose:
            print('[SemCom] Frozen: patch_embed + enc_blocks (ViT encoder)')
    elif freeze == 'none':
        if verbose:
            print('[SemCom] No freezing — full joint training')
    else:
        raise ValueError(f'Unknown freeze={freeze!r}. Choose "none" or "encoder".')

    if verbose:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f'[SemCom] Parameters: {total / 1e6:.1f}M total, '
            f'{trainable / 1e6:.1f}M trainable'
        )

    return model.to(device).train(), jscc_config
