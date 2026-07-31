#!/usr/bin/env python3
"""
DUSt3R + SemCom Interactive Demo
=================================
Gradio web UI that lets you upload images and choose:
  - Channel type (None / AWGN / Rayleigh)
  - SNR slider

The model reconstructs the 3D scene and shows how the wireless channel
degrades reconstruction quality in real time.

Usage
-----
# Noise-only baseline (no JSCC checkpoint needed)
CUDA_VISIBLE_DEVICES=1 python demo_semcom.py \\
    --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# With trained JSCC checkpoint (from train_jscc.py or train_e2e.py)
CUDA_VISIBLE_DEVICES=1 python demo_semcom.py \\
    --weights      checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
    --jscc_weights checkpoints/jscc_awgn_k512.pth
"""

import argparse
import copy
import functools
import glob
import math
import os
import tempfile

import gradio
import torch
import numpy as np
import matplotlib.pyplot as pl

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dust3r.demo import (
    _convert_scene_output_to_glb,
    get_3D_model_from_scene,
    set_scenegraph_options,
    set_print_with_timestamp,
)
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.model_semcom import load_semcom_model

pl.ion()
torch.backends.cuda.matmul.allow_tf32 = True


# ── SemCom checkpoint discovery ───────────────────────────────────────────────

_CKPT_PREFIX = 'e2eA_awgn_snr0-20_'


def tag_to_ratio(tag: str) -> float:
    """'identity' → 1.0, 'r0.125' → 0.125 (k/D compression ratio)."""
    return 1.0 if tag == 'identity' else float(tag[1:])


def discover_semcom_ckpts(jscc_dir: str, prefix: str = _CKPT_PREFIX) -> dict:
    """
    Scan `jscc_dir` for trained SemCom checkpoints laid out as
    ``<prefix><TAG>/checkpoint-last.pth`` (TAG ∈ identity, r0.5, r0.25, …).

    Returns an ordered {tag: path} dict, identity first then descending ratio,
    so one UI dropdown can switch between all trained compression ratios.
    """
    out = {}
    for p in glob.glob(os.path.join(jscc_dir, prefix + '*', 'checkpoint-last.pth')):
        tag = os.path.basename(os.path.dirname(p))[len(prefix):]
        out[tag] = p
    return {t: out[t] for t in sorted(out, key=lambda t: -tag_to_ratio(t))}


# ── Model loader ──────────────────────────────────────────────────────────────

def build_model(weights: str, device: str, channel: str, snr_db: float,
                compression: str, comp_mode: str,
                ratio_tag: str, ckpt_map: dict):
    """
    The ``ratio_tag`` dropdown (e.g. 'r0.125') drives the compression ratio for
    BOTH paths, so SemCom and the source-coding baseline are compared at the
    same k/D without restarting the server:

      compression != 'none' → source-coding baseline on the CLEAN backbone.
          'quant' uses its own bits (ratio_tag ignored); 'topk'/'token' use
          keep_frac = tag_to_ratio(ratio_tag).
          comp_mode == 'digital' → error-free delivery (channel/SNR ignored).
          comp_mode == 'analog'  → kept values sent over the SAME physical
              channel as DeepJSCC at the current SNR (bandwidth ratio = keep_frac).
      compression == 'none':
          channel == 'none'  → clean DUSt3R baseline (no SemCom noise)
          channel != 'none'  → SemCom DeepJSCC; loads the trained checkpoint
              ckpt_map[ratio_tag] at the current channel/SNR.
    """
    if compression != 'none':
        from dust3r.semcom import (UniformQuantBlock, AnalogTopKBlock, AnalogTokenPruneBlock)
        # analog source coding needs a real physical channel; default to awgn
        phys = channel if channel in ('awgn', 'rayleigh') else 'awgn'
        keep = tag_to_ratio(ratio_tag)
        model = load_semcom_model(weights, device, snr_db=float('inf'), verbose=False)
        kind, _, param = compression.partition(':')
        if kind == 'quant':
            block = UniformQuantBlock(bits=int(param))
        elif kind == 'topk':
            block = AnalogTopKBlock(keep_frac=keep, channel=phys)
        elif kind == 'token':
            block = AnalogTokenPruneBlock(keep_frac=keep, importance='norm', channel=phys)
        else:
            raise ValueError(f'unknown compression {compression!r}')
        block.snr_db = snr_db
        model.feat_semcom = block.to(device)
        return model
    if channel == 'none':
        return load_semcom_model(weights, device, snr_db=float('inf'), verbose=False)
    # SemCom DeepJSCC: load the trained checkpoint for the selected ratio.
    jscc_path = ckpt_map.get(ratio_tag)   # None → noise-only identity JSCC
    return load_semcom_model(
        weights, device,
        snr_db=snr_db, channel=channel,
        jscc_path=jscc_path, verbose=False,
    )


# ── Reconstruction function ───────────────────────────────────────────────────

def get_reconstructed_scene(
    outdir, weights, ckpt_map, device, image_size, silent,
    filelist, channel, snr_db, compression, comp_mode, ratio_tag,
    schedule, niter, min_conf_thr,
    as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
    scenegraph_type, winsize, refid,
):
    if not filelist:
        return None, None, None

    # Build model with current channel / SNR / compression / ratio settings
    model = build_model(weights, device, channel, float(snr_db),
                        compression, comp_mode, ratio_tag, ckpt_map)

    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    sg = scenegraph_type
    if sg == 'swin':
        sg = f'swin-{int(winsize)}'
    elif sg == 'oneref':
        sg = f'oneref-{int(refid)}'

    pairs = make_pairs(imgs, scene_graph=sg, prefilter=None, symmetrize=True)

    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    mode = (GlobalAlignerMode.PointCloudOptimizer
            if len(imgs) > 2 else GlobalAlignerMode.PairViewer)
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        try:
            scene.compute_global_alignment(
                init='mst', niter=int(niter), schedule=schedule, lr=0.01)
        except Exception as e:
            print(f'[GA failed: {e}]')

    outfile = get_3D_model_from_scene(
        outdir, silent, scene, min_conf_thr, as_pointcloud,
        mask_sky, clean_depth, transparent_cams, cam_size)

    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max(d.max() for d in depths)
    depths = [d / depths_max for d in depths]
    confs_max = max(d.max() for d in confs)
    confs = [cmap(d / confs_max) for d in confs]

    gallery = []
    for i in range(len(rgbimg)):
        gallery.append(rgbimg[i])
        gallery.append(rgb(depths[i]))
        gallery.append(rgb(confs[i]))

    del model
    torch.cuda.empty_cache()

    return scene, outfile, gallery


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def main_demo(tmpdirname, weights, ckpt_map, device, image_size, server_name,
              server_port, silent=False, share=False):
    recon_fn = functools.partial(
        get_reconstructed_scene,
        tmpdirname, weights, ckpt_map, device, image_size, silent,
    )

    model_from_scene_fn = functools.partial(get_3D_model_from_scene, tmpdirname, silent)

    # Ratio dropdown: every trained SemCom checkpoint found on disk + a clean
    # 'identity' option. Drives BOTH the SemCom checkpoint and topk/token frac.
    ratio_tags = list(ckpt_map.keys())
    ratio_choices = [
        (f'{t}  (k={round(tag_to_ratio(t) * 1024)}/1024'
         + (', no compression' if t == 'identity' else '') + ')', t)
        for t in ratio_tags
    ]
    default_ratio = ratio_tags[0] if ratio_tags else 'identity'
    mode_label = (f'{len(ckpt_map)} SemCom checkpoints loaded: '
                  + ', '.join(ratio_tags)) if ckpt_map else 'no JSCC found (noise-only)'

    with gradio.Blocks(title='DUSt3R × SemCom Demo') as demo:
        scene_state = gradio.State(None)

        gradio.HTML(
            '<h2 style="text-align:center;">DUSt3R × SemCom Demo</h2>'
            f'<p style="text-align:center; color:#888;">Mode: {mode_label}</p>'
        )

        with gradio.Row():
            # ── Left column: inputs ──────────────────────────────────────────
            with gradio.Column(scale=1):
                inputfiles = gradio.File(file_count='multiple', label='Input images')

                gradio.HTML('<b>📡 Channel settings</b>')
                channel = gradio.Radio(
                    choices=['none', 'awgn', 'rayleigh'],
                    value='none',
                    label='Channel type',
                    info='"none" = clean DUSt3R baseline',
                )
                snr_db = gradio.Slider(
                    label='SNR (dB)',
                    value=10.0, minimum=0.0, maximum=30.0, step=1.0,
                    info='Lower = more noise. Only used when channel ≠ none.',
                )

                gradio.HTML('<b>📐 Compression ratio (k/D)</b>')
                ratio_tag = gradio.Dropdown(
                    choices=ratio_choices, value=default_ratio,
                    label='Compression ratio',
                    info='Drives BOTH paths: SemCom loads the trained checkpoint '
                         'for this ratio, and TopK/Token use it as keep_frac — so '
                         'the two are compared at the same k/D. (Quant uses bits.)',
                )

                gradio.HTML('<b>🗜️ Source-coding baseline</b>')
                compression = gradio.Dropdown(
                    choices=[('none (use SemCom / channel above)', 'none'),
                             ('TopK (keep ratio above of dims)', 'topk'),
                             ('Token-prune (keep ratio above of tokens)', 'token'),
                             ('Quant 4-bit', 'quant:4'),
                             ('Quant 2-bit', 'quant:2'),
                             ('Quant 1-bit', 'quant:1')],
                    value='none', label='Compression (feature-domain)',
                    info='If set, overrides the SemCom JSCC path above: shows a '
                         'classical source-coding baseline on the clean DUSt3R. '
                         'TopK/Token use the ratio above; Quant uses its own bits.',
                )
                comp_mode = gradio.Radio(
                    choices=[('digital (error-free)', 'digital'),
                             ('analog (over channel @ SNR above)', 'analog')],
                    value='digital', label='Compression transmission',
                    info='digital = ideal channel code, SNR ignored (pure '
                         'compression distortion). analog = kept values sent over '
                         'the channel above at the chosen SNR (topk/token only; '
                         'quant is always digital).',
                )

                gradio.HTML('<b>⚙️ Reconstruction settings</b>')
                with gradio.Row():
                    schedule = gradio.Dropdown(
                        ['linear', 'cosine'], value='linear', label='GA schedule')
                    niter = gradio.Number(
                        value=300, precision=0, minimum=0, maximum=5000,
                        label='GA iterations')
                with gradio.Row():
                    scenegraph_type = gradio.Dropdown(
                        [('complete: all pairs', 'complete'),
                         ('swin: sliding window', 'swin'),
                         ('oneref: one vs all', 'oneref')],
                        value='complete', label='Scene graph', interactive=True)
                    winsize = gradio.Slider(
                        label='Window size', value=1, minimum=1, maximum=2,
                        step=1, visible=False)
                    refid = gradio.Slider(
                        label='Ref image id', value=0, minimum=0, maximum=1,
                        step=1, visible=False)

                run_btn = gradio.Button('▶  Run reconstruction', variant='primary')

            # ── Right column: outputs ────────────────────────────────────────
            with gradio.Column(scale=2):
                outmodel = gradio.Model3D(label='3D scene')
                with gradio.Row():
                    min_conf_thr = gradio.Slider(
                        label='min_conf_thr', value=3.0, minimum=1.0,
                        maximum=20, step=0.1)
                    cam_size = gradio.Slider(
                        label='cam_size', value=0.05, minimum=0.001,
                        maximum=0.1, step=0.001)
                with gradio.Row():
                    as_pointcloud    = gradio.Checkbox(value=False, label='Pointcloud')
                    mask_sky         = gradio.Checkbox(value=False, label='Mask sky')
                    clean_depth      = gradio.Checkbox(value=True,  label='Clean depth')
                    transparent_cams = gradio.Checkbox(value=False, label='Transparent cams')
                outgallery = gradio.Gallery(
                    label='RGB | Depth | Confidence', columns=3, height='auto')

        # ── Common input list ────────────────────────────────────────────────
        recon_inputs = [
            inputfiles, channel, snr_db, compression, comp_mode, ratio_tag,
            schedule, niter, min_conf_thr,
            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
            scenegraph_type, winsize, refid,
        ]
        viz_inputs = [
            scene_state, min_conf_thr, as_pointcloud, mask_sky,
            clean_depth, transparent_cams, cam_size,
        ]

        # ── Events ──────────────────────────────────────────────────────────
        scenegraph_type.change(
            set_scenegraph_options,
            inputs=[inputfiles, winsize, refid, scenegraph_type],
            outputs=[winsize, refid])
        inputfiles.change(
            set_scenegraph_options,
            inputs=[inputfiles, winsize, refid, scenegraph_type],
            outputs=[winsize, refid])

        run_btn.click(
            fn=recon_fn,
            inputs=recon_inputs,
            outputs=[scene_state, outmodel, outgallery])

        for component in (min_conf_thr, cam_size):
            component.release(
                fn=lambda *a: model_from_scene_fn(*a),
                inputs=viz_inputs, outputs=outmodel)

        for component in (as_pointcloud, mask_sky, clean_depth, transparent_cams):
            component.change(
                fn=lambda *a: model_from_scene_fn(*a),
                inputs=viz_inputs, outputs=outmodel)

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        ssr_mode=False,   # gradio 5/6 SSR needs node.js; without it the page hangs
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='DUSt3R + SemCom Gradio demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--weights', required=True,
                   help='DUSt3R checkpoint path.')
    p.add_argument('--jscc_dir', default='checkpoints',
                   help='Directory scanned for trained SemCom checkpoints laid '
                        f'out as {_CKPT_PREFIX}<TAG>/checkpoint-last.pth. All '
                        'discovered ratios become selectable in the UI dropdown.')
    p.add_argument('--jscc_weights', default=None,
                   help='Optional: an extra JSCC checkpoint to add to the ratio '
                        'dropdown (back-compat). Normally auto-discovered via '
                        '--jscc_dir, so this is no longer required.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--image_size', type=int, default=512, choices=[224, 512])
    p.add_argument('--server_port', type=int, default=7861,
                   help='Gradio server port.')
    p.add_argument('--local_network', action='store_true',
                   help='Bind to 0.0.0.0 (accessible from local network).')
    p.add_argument('--share', action='store_true',
                   help='Create a public gradio.live tunnel URL (bypasses local '
                        'port-forwarding, e.g. VS Code/SSH — useful if the page '
                        'hangs loading over a forwarded 127.0.0.1 port).')
    p.add_argument('--silent', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    set_print_with_timestamp()

    ckpt_map = discover_semcom_ckpts(args.jscc_dir)
    if args.jscc_weights:  # back-compat: make an explicitly-passed ckpt selectable
        tag = os.path.basename(os.path.dirname(args.jscc_weights))
        tag = tag[len(_CKPT_PREFIX):] if tag.startswith(_CKPT_PREFIX) else tag
        ckpt_map.setdefault(tag, args.jscc_weights)
    if ckpt_map:
        print(f'[SemCom] {len(ckpt_map)} checkpoints selectable: '
              + ', '.join(ckpt_map))
    else:
        print(f'[SemCom] no checkpoints found under {args.jscc_dir!r} '
              f'(pattern {_CKPT_PREFIX}*/checkpoint-last.pth) — '
              'SemCom path will run noise-only identity JSCC.')

    server_name = '0.0.0.0' if args.local_network else '127.0.0.1'
    with tempfile.TemporaryDirectory(suffix='_semcom_demo') as tmpdirname:
        main_demo(
            tmpdirname=tmpdirname,
            weights=args.weights,
            ckpt_map=ckpt_map,
            device=args.device,
            image_size=args.image_size,
            server_name=server_name,
            server_port=args.server_port,
            silent=args.silent,
            share=args.share,
        )
