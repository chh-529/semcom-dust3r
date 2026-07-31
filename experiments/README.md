# Experiment Configs

Each `.yaml` file in this directory defines one study. Run with:

```bash
python run_exp.py experiments/<study>.yaml [--dry-run] [--gpus 1 2]
```

## YAML Schema

```yaml
name:        <str>          # unique study ID (used for log dir)
description: <str>          # human-readable description
type:        eval | train   # determines CLI and resume logic
script:      <str>          # Python script name without .py (must be in repo root)
output_dir:  <str>          # where outputs are written  (eval only)
python:      <str>          # full path to python interpreter
gpus:        [int, …]       # default GPU pool

env:                        # extra environment variables
  KEY: VALUE

defaults:                   # args applied to every job
  weights: checkpoints/...
  n_views: 20
  gt_pose: true             # boolean true → --gt_pose flag

sweep:                      # cartesian product of axes
  - axis: snr               # scalar axis: axis name = CLI arg name
    values: [0, 5, 10, 20]
  - axis: model             # named-group axis: each value is a param dict
    values:
      - label: r0.5         # 'label' is internal (naming only, not a CLI arg)
        jscc_weights: checkpoints/e2eA_awgn_snr0-20_r0.5/checkpoint-last.pth

output: "{label}_snr{snr}.json"   # template for output filename (eval)

extra:                      # one-off jobs not covered by the sweep
  - label: clean
    snr: inf
    output: clean_snrinf.json     # explicit filename overrides template
```

### train type
For `type: train`, set `output_dir` to a template (e.g. `checkpoints/e2eA_r{ratio}/`).
The runner creates the directory, passes `--output_dir`, and auto-appends
`--resume checkpoint-last.pth` if a checkpoint already exists.

## Studies

| File | Type | Description |
|------|------|-------------|
| `dtu_arch_a_matrix.yaml`    | eval  | Arch A × SNR × rate — main result table |
| `dtu_source_coding.yaml`    | eval  | Digital baselines (quant / topk / token / JPEG) |
| `dtu_analog_full22.yaml`    | eval  | Analog TopK / Token vs learned (full 22 scans) |
| `dtu_view_sweep.yaml`       | eval  | View-count sweep (SemCom) |
| `dtu_arch_b_snr.yaml`       | eval  | Arch B image-domain JSCC × SNR |
| `dtu_cr_nv6.yaml`           | eval  | CR sweep at n_views=6 (Arch B + JPEG budget) |
| `train/arch_a.yaml`         | train | Arch A end-to-end training (rate sweep) |
| `train/arch_b.yaml`         | train | Arch B end-to-end training (c_out sweep) |
