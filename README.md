# DUSt3R + Semantic Communication (minimal)

A minimal, self-contained version of the DUSt3R × Semantic Communication (SemCom)
study: a learned joint source-channel coding (JSCC) block inserted into the
DUSt3R 3D-reconstruction pipeline, so that 3D reconstruction happens *across a
noisy wireless channel*.

This repo contains everything needed to **train** and **evaluate** the
architecture. It deliberately excludes the original study's experiment sweeps,
plotting scripts and result archives.

---

## The two architectures

```
Architecture A — feature domain          (model.feat_semcom)
  Tx │ Image → [ViT Encoder] → feat (B,N,1024)
     │        → [JSCC Enc · MLP 1024→k] → power-norm → ┐
     │                                                 │  AWGN / Rayleigh
  Rx │ ┌───────────────────────────────────────────────┘
     │ └→ [JSCC Dec · MLP k→1024] → feat_hat → [Cross-view Decoder] → [Head] → pts3d

Architecture B — image domain            (model.img_semcom)
  Tx │ Image (3,H,W) → [CNN Enc · 5 layers, 4× down → c_out ch] → power-norm → ┐
     │                                                                         │
  Rx │ ┌───────────────────────────────────────────────────────────────────────┘
     │ └→ [CNN Dec → tanh] → img_hat → [ViT Enc + Cross-view Decoder + Head] → pts3d
```

**Rate accounting.** Arch A sends `k·H·W/256` real symbols per image
(compression ratio `k/1024`); Arch B sends `c_out·H·W/16` (bandwidth ratio
`c_out/48`). To compare the two at an equal channel budget, set

```
c_out = k / 16          e.g.  k = 128  ↔  c_out = 8
```

Setting both blocks to `None` gives a clean DUSt3R baseline (no channel).

---

## Layout

```
dust3r/
  semcom/               ← the SemCom contribution
    channel.py            AWGNChannel, RayleighChannel
    feature_jscc.py       Arch A: JSCCEncoder/Decoder (MLP) + SemComBlock
    image_jscc.py         Arch B: DeepJSCC CNN encoder/decoder + ImageSemComBlock
    source_coding.py      Baselines: uniform quant, JPEG/WebP, analog TopK, token prune
  model_semcom.py       ← DUSt3RSemCom wrapper, transmit-once inference, build/load
  ...                   ← upstream DUSt3R (unmodified except 3 compat patches, see below)
croco/                  ← upstream CroCo backbone (unmodified)

train_e2e.py            End-to-end joint training (Arch A and Arch B)
eval_dtu.py             DTU multi-view eval: global alignment → Acc/Comp/Overall (mm)
eval_semcom.py          BlendedMVS pairwise eval (no external deps beyond the dataset)
demo_semcom.py          Interactive Gradio demo
run_exp.py              YAML-driven sweep runner (multi-GPU, resumable)
experiments/            Example YAML configs
tools/test_tx_once.py   Sanity check for the transmit-once path
datasets_preprocess/    BlendedMVS preprocessing (for training)
```

### Where to plug in your own ideas

| You want to change | Edit |
|---|---|
| Channel model (fading, MIMO, …) | `dust3r/semcom/channel.py` — add a class with `forward(z, snr_db)` |
| Arch A codec architecture | `dust3r/semcom/feature_jscc.py` — `JSCCEncoder` / `JSCCDecoder` |
| Arch B codec architecture | `dust3r/semcom/image_jscc.py` — `ImageJSCCEncoder` / `ImageJSCCDecoder` |
| A new compression baseline | `dust3r/semcom/source_coding.py` |
| Where the channel sits | `dust3r/model_semcom.py` — `DUSt3RSemCom.forward` |

**All channel blocks share one interface**, so any of them can be dropped into
`model.feat_semcom` without touching the eval pipeline:

```python
forward(feat_or_img, snr_db=None) -> same shape
```

Keep that contract and everything downstream keeps working.

---

## Setup

```bash
git clone <this-repo> dust3r-semcom && cd dust3r-semcom

conda create -n dust3r python=3.11
conda activate dust3r
pip install torch torchvision          # match your CUDA version
pip install -r requirements.txt

# eval_dtu.py additionally needs:
pip install open3d scikit-learn
```

Optional — compile the CroCo RoPE CUDA kernel for a speedup (works without it):

```bash
cd croco/models/curope && python setup.py build_ext --inplace && cd ../../..
```

### Checkpoints

`checkpoints/` is gitignored — weights are not in this repo. On the lab machine
they can be copied straight from the shared directory:

```bash
export SHARED=/tmp2/b12902145/dust3r/checkpoints
mkdir -p checkpoints
```

**1. DUSt3R backbone — required for everything** (2.2 GB)

```bash
cp $SHARED/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth checkpoints/
```

Or download it from upstream:

```bash
wget -P checkpoints https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

**2. Trained SemCom models — optional**, only if you want to evaluate rather
than train from scratch. Each is ~6.4 GB (2.3 GB of weights + optimizer state),
so copy just the ones you need:

```bash
cp -r $SHARED/e2eA_awgn_snr0-20_r0.125 checkpoints/     # Arch A, k/1024 = 0.125
cp -r $SHARED/e2eB_awgn_snr0-20_c8     checkpoints/     # Arch B, budget-matched
```

| Arch A (`--domain feature`) | ratio `k/1024` | Arch B (`--domain image`) | ratio `c_out/48` |
|---|---|---|---|
| `e2eA_awgn_snr0-20_r0.5`    | 0.5    | `e2eB_awgn_snr0-20_c24` | 0.500 |
| `e2eA_awgn_snr0-20_r0.25`   | 0.25   | `e2eB_awgn_snr0-20_c12` | 0.250 |
| `e2eA_awgn_snr0-20_r0.125`  | 0.125  | `e2eB_awgn_snr0-20_c8`  | 0.167 |
| `e2eA_awgn_snr0-20_r0.083`  | 0.083  | `e2eB_awgn_snr0-20_c6`  | 0.125 |
| `e2eA_awgn_snr0-20_r0.0625` | 0.0625 | `e2eB_awgn_snr0-20_c4`  | 0.083 |
| `e2eA_awgn_snr0-20_identity`| 1.0 (noise only, no codec) | `e2eB_awgn_snr0-20_c{1,2,3,48}` | 0.021 … 1.0 |

All were trained on BlendedMVS for 20–50 epochs with the SNR sampled uniformly
in [0, 20] dB, so a single checkpoint can be evaluated at any SNR — pass
`--snr` at eval time and it overrides whatever the checkpoint was trained at.

Pairs on the same row are **budget-matched** (`c_out = k/16`), which is the
comparison the two architectures are meant to be read against.

### Datasets

| Purpose | Dataset | Notes |
|---|---|---|
| Training | BlendedMVS | preprocess with `datasets_preprocess/preprocess_blendedMVS.py` → `data/blendedmvs_processed` |
| Eval (main) | DTU test set + GT point clouds | also needs a clone of [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) |
| Eval (light) | BlendedMVS val split | reuses the training preprocessing; no extra tools |

---

## Quick start

**1. Does it run at all?** (only needs the backbone checkpoint)

```bash
python demo_semcom.py --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

**2. Train Arch A** (feature-domain JSCC, k = 0.125 × 1024 = 128)

```bash
python train_e2e.py \
    --weights   checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    --dataset   "10000 @ BlendedMVS(split='train', ROOT='data/blendedmvs_processed', resolution=512, aug_crop=16)" \
    --domain    feature --ratio 0.125 \
    --channel   awgn --snr_range 0 20 \
    --epochs    50 --batch_size 2 --accum_iter 8 \
    --lr 5e-5 --backbone_lr_scale 0.05 --amp \
    --output_dir checkpoints/e2eA_awgn_snr0-20_r0.125/
```

**3. Train Arch B** (image-domain JSCC, budget-matched: `c_out = 128/16 = 8`)

```bash
python train_e2e.py \
    --weights   checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    --dataset   "10000 @ BlendedMVS(split='train', ROOT='data/blendedmvs_processed', resolution=512, aug_crop=16)" \
    --domain    image --img_c_out 8 \
    --channel   awgn --snr_range 0 20 \
    --epochs    20 --batch_size 1 --accum_iter 8 \
    --lr 5e-5 --backbone_lr_scale 0.05 --amp \
    --output_dir checkpoints/e2eB_awgn_snr0-20_c8/
```

The architecture is recorded in the checkpoint, so evaluation auto-detects
whether a checkpoint is Arch A or Arch B — you never pass `--domain` at eval time.

**4. Evaluate on DTU**

```bash
python eval_dtu.py \
    --weights      checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    --jscc_weights checkpoints/e2eA_awgn_snr0-20_r0.125/checkpoint-last.pth \
    --snr 10 --tx_mode once --n_views 20 --gt_pose \
    --dtu_test /path/to/dtu/dtu-test \
    --gt_dir   /path/to/dtu \
    --dtueval  /path/to/DTUeval-python \
    --output   results/r0.125_snr10.json
```

Baselines, all through the same script:

```bash
# clean DUSt3R (no channel)
--snr inf                      # and drop --jscc_weights

# noise-only, no learned codec (identity JSCC at full bandwidth)
--snr 10                       # and drop --jscc_weights

# digital source coding (assumes error-free delivery — separation-theorem bound)
--compression quant:4          # 4 bits/dim
--compression jpeg:50          # JPEG quality 50

# unlearned analog compression over the SAME physical channel
--compression topk:0.125 --analog
--compression token:0.25 --analog
```

**5. Sweeps** — instead of shell loops, use `run_exp.py`:

```bash
python run_exp.py experiments/train/arch_a.yaml --dry-run   # preview jobs
python run_exp.py experiments/train/arch_a.yaml --gpus 0
python run_exp.py experiments/eval_dtu_example.yaml --gpus 0 1
```

Jobs are resumable: an eval whose output JSON already has a `mean` key is
skipped; a training run with an existing `checkpoint-last.pth` is resumed.
See [experiments/README.md](experiments/README.md) for the schema.

---

## One thing to get right: `--tx_mode`

```
per_pair  every image is re-transmitted, with fresh noise, for every pair it
          appears in.  Noise averages out across the scene graph — this is free
          retransmission diversity and it makes the channel look far too benign.

once      each image passes through the channel exactly once; the received
          features are cached and reused by every pair.  This is what a real
          system does, and channel errors stay correlated across the scene.
```

**Use `--tx_mode once` for anything you intend to report.** `per_pair` is the
default only for backward compatibility with the upstream inference path.

---

## Relationship to upstream DUSt3R

The SemCom integration subclasses rather than edits: `DUSt3RSemCom` extends
`AsymmetricCroCo3DStereo` and overrides `forward()` only. Upstream DUSt3R was
touched in exactly three places, all compatibility fixes rather than behaviour
changes:

| File | Change | Why |
|---|---|---|
| `dust3r/model.py` | `torch.load(..., weights_only=False)` | torch ≥ 2.6 changed the default |
| `dust3r/cloud_opt/base_opt.py` | `torch.amp.autocast('cuda', ...)` | `torch.cuda.amp.autocast` deprecated |
| `dust3r/demo.py` | gradio slider bounds, css argument | gradio 6.x compatibility |

Keeping these separate means you can rebase onto a newer DUSt3R with little friction.

---

## License

DUSt3R and CroCo are © Naver Corporation, CC BY-NC-SA 4.0 (non-commercial).
See [LICENSE](LICENSE) and [NOTICE](NOTICE). The SemCom additions under
`dust3r/semcom/` and `dust3r/model_semcom.py` inherit the same terms.
