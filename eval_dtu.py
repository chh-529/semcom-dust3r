#!/usr/bin/env python3
"""
DTU Reconstruction Evaluation for DUSt3R + SemCom
==================================================
Cross-domain (held-out) 3D reconstruction evaluation on the DTU MVS benchmark,
mirroring the DUSt3R paper's Table 4 (Acc / Comp / Overall in mm).

Pipeline (per scan)
-------------------
  images ─▶ DUSt3R+SemCom inference (JSCC at a chosen SNR)
         ─▶ global alignment (PointCloudOptimizer)  ─▶ fused point cloud
         ─▶ align to DTU world frame + mm scale  (Umeyama on camera centres
            vs GT cam centres — DUSt3R is pose/scale-free, GT cams ✗)
         ─▶ export .ply
         ─▶ jzhangbs/DTUeval-python  ─▶ Acc / Comp / Overall (mm)

Why Umeyama on camera centres?  DTUeval masks the prediction with ObsMask (a 3D
grid in DTU world coordinates) and measures mm distances, so the prediction must
live in the DTU world frame at metric scale.  DUSt3R does not use GT poses during
inference, so we register its reconstruction to GT afterwards using the GT camera
centres as correspondences (a similarity transform: scale + rotation + translation).

Usage
-----
# Clean DUSt3R upper bound (no channel), a few scans:
python eval_dtu.py \
    --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    --dtu_test  /path/to/dtu/dtu-test \
    --gt_dir    /path/to/dtu \
    --dtueval   /path/to/DTUeval-python \
    --snr inf --scans 1 4 9 --n_views 15 \
    --output results/dtu_clean.json

# E2E 1/8 at SNR=10 dB, all 22 scans:
python eval_dtu.py ... --jscc_weights checkpoints/e2eA_awgn_snr0-20_r0.125/checkpoint-last.pth \
    --channel awgn --snr 10 --output results/dtu_e2e_r0.125_snr10.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dust3r.model_semcom import load_semcom_model, run_inference
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# Standard 22 DTU evaluation scans (MVSNet test split).
DTU_SCANS = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48,
             49, 62, 75, 77, 110, 114, 118]


# ── Camera helpers ──────────────────────────────────────────────────────────────

def read_mvsnet_extrinsic(cam_path: str) -> np.ndarray:
    """Parse an MVSNet `*_cam.txt` file → 4x4 world2cam extrinsic."""
    with open(cam_path) as f:
        toks = f.read().split()
    i = toks.index('extrinsic')
    return np.array(toks[i + 1:i + 17], dtype=np.float64).reshape(4, 4)


def cam_centre_world2cam(extrinsic: np.ndarray) -> np.ndarray:
    """Camera centre in world coords from a world2cam extrinsic: C = -R^T t."""
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    return -R.T @ t


def umeyama(src: np.ndarray, dst: np.ndarray):
    """
    Similarity transform aligning src→dst (Umeyama 1991):  dst ≈ s·R·src + t.
    src, dst : (N, 3).  Returns (s, R, t).
    """
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


# ── Reconstruction ──────────────────────────────────────────────────────────────

def _subsample(items, n, rng=None):
    if n <= 0 or n >= len(items):
        return list(range(len(items)))
    if rng is not None:
        return sorted(rng.choice(len(items), n, replace=False).tolist())
    return sorted(set(np.linspace(0, len(items) - 1, n).round().astype(int).tolist()))


def reconstruct_scan(model, scan_dir, device, n_views, niter, conf_thr, snr_db,
                     image_size, gt_pose=False, silent=True, tx_mode='per_pair',
                     view_list=None, seed=None, with_color=False):
    """
    Run DUSt3R+SemCom over a DTU scan.

    gt_pose=False (default): pose-free reconstruction.  Returns
        (cloud, pred_centres, gt_centres) for post-hoc Umeyama alignment.
    gt_pose=True (paper protocol): GT camera poses are FIXED as constants in the
        global alignment (scene.preset_pose), so the reconstruction is produced
        directly in the DTU world frame at metric scale — no Umeyama needed.
        Returns (cloud_in_gt_frame, None, None).

    tx_mode='per_pair': each pair forward re-transmits both views through the
        channel (independent noise per pair — retransmission diversity).
    tx_mode='once': each view's features pass through the channel exactly once
        and are reused by all pairs (correlated errors — physical transmit-once).

    view_list: optional list of filenames (e.g. ['00000002.jpg', ...]) to use
        instead of _subsample.  Overrides n_views.

    with_color: if True, also return per-point RGB (float [0,1], same masking
        as cloud) sourced from the INPUT images (not touched by the channel
        in either architecture -- see eval_color_psnr.py). Appended as an
        extra return value so existing callers are unaffected.
    """
    img_dir = os.path.join(scan_dir, 'images')
    cam_dir = os.path.join(scan_dir, 'cams')
    img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png')))
    if view_list is not None:
        idxs = [img_files.index(f) for f in view_list]
    else:
        rng = np.random.default_rng(seed) if seed is not None else None
        idxs = _subsample(img_files, n_views, rng=rng)
    filelist = [os.path.join(img_dir, img_files[i]) for i in idxs]
    camfile = lambda i: os.path.join(cam_dir, os.path.splitext(img_files[i])[0] + '_cam.txt')

    if model.feat_semcom is not None:
        model.feat_semcom.snr_db = snr_db

    imgs = load_images(filelist, size=image_size, verbose=not silent,
                       patch_size=model.patch_size)
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    with torch.no_grad():
        if tx_mode == 'once':
            output = run_inference(pairs, model, device, batch_size=4,
                                   verbose=not silent)
        else:
            output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    # NOTE: global alignment is a gradient-descent optimisation over poses/depths,
    # so it must run WITH grad enabled (do not wrap in torch.no_grad()).
    scene = global_aligner(output, device=device,
                           mode=GlobalAlignerMode.PointCloudOptimizer, verbose=not silent)

    if gt_pose:
        # Paper protocol: anchor to GT by fixing GT camera poses (cam2world).
        gt_c2w = np.stack([np.linalg.inv(read_mvsnet_extrinsic(camfile(i))) for i in idxs])
        scene.preset_pose(torch.tensor(gt_c2w, dtype=torch.float32, device=device))

    scene.compute_global_alignment(init='mst', niter=niter, schedule='cosine', lr=0.01)
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(conf_thr)))

    pts3d = to_numpy(scene.get_pts3d())
    masks = to_numpy(scene.get_masks())
    cloud = np.concatenate([p[m] for p, m in zip(pts3d, masks)], axis=0).astype(np.float64)
    colors = None
    if with_color:
        colors = np.concatenate([scene.imgs[i][m] for i, m in enumerate(masks)],
                                axis=0).astype(np.float64)

    if gt_pose:
        return (cloud, None, None, colors) if with_color else (cloud, None, None)

    poses = to_numpy(scene.get_im_poses())           # (V,4,4) cam2world
    pred_centres = poses[:, :3, 3]
    gt_centres = np.stack([cam_centre_world2cam(read_mvsnet_extrinsic(camfile(i)))
                           for i in idxs])
    if with_color:
        return cloud, pred_centres.astype(np.float64), gt_centres, colors
    return cloud, pred_centres.astype(np.float64), gt_centres


def align_cloud(cloud, pred_centres, gt_centres):
    """Register DUSt3R cloud into DTU world+mm via Umeyama on camera centres."""
    s, R, t = umeyama(pred_centres, gt_centres)
    return (s * (cloud @ R.T)) + t, (s, R, t)


# ── DTUeval-python wrapper ──────────────────────────────────────────────────────

# DTUeval caps each point's distance at max_dist=20mm; a fully collapsed
# reconstruction (≈0 predicted points inside the ObsMask volume) is "worse than
# measurable", so we score it at this cap rather than dropping it. Dropping
# collapsed scans would FLATTER methods that fail on hard scenes (survivorship
# bias); counting them at the penalty keeps the cross-method comparison fair.
COLLAPSE_PENALTY = 20.0


def run_dtueval(ply_path, scan, gt_dir, dtueval_dir, vis_dir, python_bin):
    """Call DTUeval-python eval.py; return (acc, comp, overall, collapsed)."""
    cmd = [python_bin, os.path.join(dtueval_dir, 'eval.py'),
           '--data', ply_path, '--scan', str(scan), '--mode', 'pcd',
           '--dataset_dir', gt_dir, '--vis_out_dir', vis_dir]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Empty masked cloud → sklearn NearestNeighbors raises on 0 samples.
        if '0 sample' in res.stderr or 'minimum of 1 is required' in res.stderr:
            return COLLAPSE_PENALTY, COLLAPSE_PENALTY, COLLAPSE_PENALTY, True
        raise RuntimeError(f'DTUeval failed (scan {scan}):\n{res.stderr[-2000:]}')
    last = [l for l in res.stdout.strip().splitlines() if l.strip()][-1]
    acc, comp, overall = (float(x) for x in last.split())
    return acc, comp, overall, False


# ── Main ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='DTU reconstruction eval for DUSt3R+SemCom',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--weights', required=True, help='DUSt3R backbone .pth')
    p.add_argument('--jscc_weights', default=None,
                   help='JSCC/E2E checkpoint; omit for noise-only/clean baseline.')
    p.add_argument('--channel', default='awgn', choices=['awgn', 'rayleigh'])
    p.add_argument('--snr', default='inf',
                   help='Inference SNR in dB ("inf" for noiseless).')
    p.add_argument('--dtu_test', required=True, help='dir with scanN/images + cams')
    p.add_argument('--gt_dir', required=True,
                   help='dir containing Points/ and ObsMask/ (for DTUeval)')
    p.add_argument('--dtueval', required=True, help='clone of jzhangbs/DTUeval-python')
    p.add_argument('--dtueval_python', default=sys.executable,
                   help='python with open3d+sklearn+scipy for DTUeval-python')
    p.add_argument('--scans', type=int, nargs='+', default=DTU_SCANS)
    p.add_argument('--gt_pose', action='store_true',
                   help='Paper protocol: fix GT camera poses in global alignment so the '
                        'reconstruction lands directly in DTU world+mm (no Umeyama). '
                        'Strongly recommended — matches DUSt3R Table 4 alignment.')
    p.add_argument('--tx_mode', default='per_pair', choices=['per_pair', 'once'],
                   help='"per_pair": re-transmit features through the channel for every '
                        'pair (independent noise). "once": transmit each view once and '
                        'reuse the received features for all pairs (physical setting).')
    p.add_argument('--n_views', type=int, default=15,
                   help='views per scan to reconstruct (0=all ~49). More=heavier.')
    p.add_argument('--niter', type=int, default=300, help='global-alignment iters')
    p.add_argument('--conf_thr', type=float, default=1.5, help='min confidence')
    p.add_argument('--image_size', type=int, default=512, choices=[224, 512])
    p.add_argument('--device', default='cuda')
    p.add_argument('--ply_dir', default=None,
                   help='where to keep aligned prediction .ply (default: temp)')
    p.add_argument('--compression', default='none',
                   help='Source-coding baseline on the CLEAN DUSt3R (no JSCC '
                        'checkpoint). Format "<kind>:<param>": quant:<bits>, '
                        'topk:<frac>, token:<frac>, jpeg:<quality>. Mutually '
                        'exclusive with --jscc_weights. "none" → plain '
                        'JSCC/clean model.')
    p.add_argument('--analog', action='store_true',
                   help='Send the compressed features over the SAME physical '
                        'channel as DeepJSCC (at --snr), instead of assuming '
                        'error-free digital delivery. Only valid for topk/token '
                        '(quant/jpeg are inherently digital). Lets the unlearned '
                        'analog compressor be compared with JSCC on the SNR axis.')
    p.add_argument('--output', required=True, help='result JSON path')
    p.add_argument('--view_list', default=None,
                   help='Comma-separated filenames to use instead of --n_views subsampling '
                        '(e.g. "00000002.jpg,00000010.jpg,..."). Overrides --n_views.')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed for view selection. Each scan uses seed+scan_id so '
                        'different scans get independent random subsets. '
                        'If omitted, uses deterministic uniform subsampling (linspace).')
    return p.parse_args()


def build_compression_block(spec: str, analog: bool = False, channel: str = 'awgn'):
    """Parse "<kind>:<param>" → (block, slot) where slot is 'feat_semcom'|'img_semcom'."""
    from dust3r.semcom import (UniformQuantBlock, ImageCodecBlock, BudgetJPEGBlock,
                               AnalogTopKBlock, AnalogTokenPruneBlock)
    kind, _, param = spec.partition(':')
    if kind == 'quant':
        return UniformQuantBlock(bits=int(param)), 'feat_semcom'
    if kind == 'topk':
        return AnalogTopKBlock(keep_frac=float(param), channel=channel), 'feat_semcom'
    if kind == 'token':
        return AnalogTokenPruneBlock(keep_frac=float(param), importance='norm',
                                     channel=channel), 'feat_semcom'
    if kind == 'jpeg':
        return ImageCodecBlock(codec='jpeg', quality=int(param)), 'img_semcom'
    if kind == 'jpeg_budget':
        bw_ratio = float(param) if param else 8 / 48.0
        return BudgetJPEGBlock(bandwidth_ratio=bw_ratio), 'img_semcom'
    raise ValueError(f'Unknown compression spec {spec!r}')


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    snr_db = float('inf') if str(args.snr).lower() == 'inf' else float(args.snr)

    if args.compression != 'none' and args.jscc_weights:
        raise SystemExit('--compression and --jscc_weights are mutually exclusive.')

    print('\n' + '=' * 64)
    print('  DUSt3R × SemCom — DTU Reconstruction Eval')
    print('=' * 64)
    print(f'  jscc   : {args.jscc_weights or "noise-only/clean"}')
    print(f'  compr  : {args.compression}')
    print(f'  channel: {args.channel}   SNR: {args.snr}   tx_mode: {args.tx_mode}')
    print(f'  scans  : {args.scans}')
    print(f'  views  : {args.n_views}/scan   niter: {args.niter}')
    print('=' * 64 + '\n')

    model = load_semcom_model(args.weights, device, snr_db=snr_db,
                              channel=args.channel, jscc_path=args.jscc_weights,
                              verbose=True)

    comp_block = None
    if args.compression != 'none':
        comp_block, slot = build_compression_block(
            args.compression, analog=args.analog, channel=args.channel)
        setattr(model, slot, comp_block.to(device))
        comp_block.snr_db = snr_db
        print(f'[compression] {args.compression} on {slot}  snr_db={snr_db}')

    auto_ply = args.ply_dir is None
    ply_dir = args.ply_dir or tempfile.mkdtemp(prefix='dtu_pred_')
    os.makedirs(ply_dir, exist_ok=True)
    vis_dir = os.path.join(ply_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    per_scan = {}
    for scan in args.scans:
        scan_dir = os.path.join(args.dtu_test, f'scan{scan}')
        if not os.path.isdir(scan_dir):
            print(f'  [skip] scan{scan}: not found at {scan_dir}')
            continue
        print(f'  ── scan{scan} ─────────────────────────────', flush=True)
        view_list = [v.strip() for v in args.view_list.split(',')] \
            if args.view_list else None
        try:
            per_scan_seed = args.seed + scan if args.seed is not None else None
            cloud, pc, gc = reconstruct_scan(
                model, scan_dir, device, args.n_views, args.niter,
                args.conf_thr, snr_db, args.image_size, gt_pose=args.gt_pose,
                tx_mode=args.tx_mode, view_list=view_list, seed=per_scan_seed)
            if args.gt_pose:
                cloud_aln, s = cloud, 1.0          # already in DTU world+mm
            else:
                cloud_aln, (s, _, _) = align_cloud(cloud, pc, gc)
            if cloud.shape[0] == 0:
                # Reconstruction produced no points at all → max-penalty collapse.
                acc = comp = overall = COLLAPSE_PENALTY
                collapsed = True
            else:
                ply_path = os.path.join(ply_dir, f'scan{scan}_pred.ply')
                trimesh.PointCloud(cloud_aln).export(ply_path)
                acc, comp, overall, collapsed = run_dtueval(
                    ply_path, scan, args.gt_dir, args.dtueval, vis_dir, args.dtueval_python)
            per_scan[scan] = {'acc': acc, 'comp': comp, 'overall': overall,
                              'collapsed': collapsed,
                              'scale': float(s), 'n_pts': int(cloud.shape[0])}
            tag = '  [COLLAPSED→penalty]' if collapsed else ''
            print(f'     Acc={acc:.4f}  Comp={comp:.4f}  Overall={overall:.4f} mm '
                  f'(scale={s:.3g}, {cloud.shape[0]} pts){tag}')
        except Exception as e:
            print(f'     [error] scan{scan}: {e}')
            per_scan[scan] = {'error': str(e)}

    # ── Summary ──────────────────────────────────────────────────────────────
    ok = {k: v for k, v in per_scan.items() if 'overall' in v}
    summary = {}
    if ok:
        # nanmean: a single bad scan shouldn't nuke the whole-set mean.
        summary = {m: float(np.nanmean([v[m] for v in ok.values()]))
                   for m in ('acc', 'comp', 'overall')}
        summary['n_scans'] = len(ok)
        summary['n_collapsed'] = sum(v.get('collapsed', False) for v in ok.values())
        print('\n' + '=' * 64)
        print(f'  MEAN over {len(ok)} scans:  '
              f'Acc={summary["acc"]:.4f}  Comp={summary["comp"]:.4f}  '
              f'Overall={summary["overall"]:.4f} mm  '
              f'({summary["n_collapsed"]} collapsed→penalty)')
        print('=' * 64)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'jscc_weights': args.jscc_weights,
            'compression': args.compression,
            'compression_rate': comp_block.rate_summary() if comp_block is not None else None,
            'channel': args.channel,
            'snr': args.snr,
            'tx_mode': args.tx_mode,
            'n_views': args.n_views,
            'seed': args.seed,
            'niter': args.niter,
            'mean': summary,
            'per_scan': per_scan,
        }, f, indent=2)
    print(f'\n  [Saved] {args.output}')

    # Clean up the auto-created temp ply/vis dir (millions of points/scan)
    # so repeated runs don't fill the disk.  User-provided --ply_dir is kept.
    if auto_ply:
        import shutil
        shutil.rmtree(ply_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
