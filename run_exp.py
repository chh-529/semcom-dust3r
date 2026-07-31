#!/usr/bin/env python3
"""
run_exp.py — universal YAML-driven experiment scheduler.

Reads an experiment YAML config, expands the parameter sweep into a flat job
list, and dispatches jobs across one or more GPUs with greedy scheduling.
Resumable: eval jobs whose output JSON already contains a ``mean`` key are
skipped; train jobs whose ``output_dir/checkpoint-last.pth`` already exists
are resumed (passed ``--resume``).

Usage
-----
    python run_exp.py experiments/dtu_arch_a_matrix.yaml
    python run_exp.py experiments/dtu_arch_a_matrix.yaml --dry-run
    python run_exp.py experiments/dtu_arch_a_matrix.yaml --gpus 0 1 2
    python run_exp.py experiments/train/arch_a.yaml --gpus 1

YAML schema
-----------
See experiments/README.md for the full schema reference.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Keys that control the runner but are NOT forwarded as CLI args.
_INTERNAL_KEYS = frozenset({"label", "output"})


# ── YAML loading ──────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Sweep expansion ───────────────────────────────────────────────────────────

def expand_sweep(axes: list) -> list[dict]:
    """
    Expand a list of sweep-axis dicts into the cartesian product.

    Each axis is:
        {axis: 'snr',   values: [0, 5, 10, 20]}          → scalar axis
        {axis: 'model', values: [{label: 'r0.5', ...}, …]} → named-group axis

    Returns a list of merged param dicts (one per job).
    """
    if not axes:
        return [{}]

    per_axis = []
    for ax in axes:
        name = ax["axis"]
        vals = ax["values"]
        if vals and isinstance(vals[0], dict):
            per_axis.append(vals)          # each dict is one point on this axis
        else:
            per_axis.append([{name: v} for v in vals])

    combos = []
    for combo in itertools.product(*per_axis):
        merged: dict = {}
        for d in combo:
            merged.update(d)
        combos.append(merged)
    return combos


# ── CLI command building ──────────────────────────────────────────────────────

def params_to_args(params: dict) -> list[str]:
    """Convert a param dict to a flat CLI arg list."""
    args: list[str] = []
    for k, v in params.items():
        if k in _INTERNAL_KEYS:
            continue
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
        elif v is None:
            pass
        elif isinstance(v, list):
            args.extend([f"--{k}"] + [str(x) for x in v])
        else:
            args.extend([f"--{k}", str(v)])
    return args


# ── Resumability checks ───────────────────────────────────────────────────────

def eval_is_done(path: Path) -> bool:
    """Return True if the eval output JSON already has a 'mean' key."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path) as f:
            d = json.load(f)
        return bool(d.get("mean"))
    except Exception:
        return False


def train_is_done(output_dir: Path) -> bool:
    """Return True if a training checkpoint already exists."""
    return (output_dir / "checkpoint-last.pth").exists()


# ── Job output path resolution ────────────────────────────────────────────────

def resolve_output(cfg: dict, params: dict) -> Path:
    """
    For eval jobs: output_dir / output_template.format(**params).
    For train jobs: output_dir_template.format(**params).
    """
    if cfg["type"] == "train":
        tpl = cfg.get("output_dir", "checkpoints/{label}/")
        return Path(tpl.format(**params))
    else:
        tpl = cfg["output"]
        filename = tpl.format(**{k: str(v) for k, v in params.items()})
        return Path(cfg["output_dir"]) / filename


# ── Main scheduler ────────────────────────────────────────────────────────────

def run_all(cfg: dict, gpus: list[int], dry_run: bool = False) -> None:
    exp_type = cfg.get("type", "eval")   # eval | train
    script   = cfg["script"]
    python   = cfg.get("python", sys.executable)
    log_root = Path("logs") / cfg["name"]

    # Build job list
    sweep_axes  = cfg.get("sweep", [])
    extra_jobs  = cfg.get("extra", [])
    defaults    = cfg.get("defaults", {})

    combos = expand_sweep(sweep_axes)
    all_param_sets = [dict(**defaults, **c) for c in combos] + \
                     [dict(**defaults, **e) for e in extra_jobs]

    jobs = []
    for params in all_param_sets:
        out = resolve_output(cfg, params)
        # Extra jobs may embed their own 'output' key as a plain filename
        if "output" in params and exp_type == "eval":
            out = Path(cfg["output_dir"]) / params["output"]
        jobs.append({"params": params, "output": out})

    print(f"Study : {cfg['name']}")
    print(f"Script: {script}.py   Type: {exp_type}   GPUs: {gpus}")
    print(f"Jobs  : {len(jobs)} total")

    if dry_run:
        for j in jobs:
            done = eval_is_done(j["output"]) if exp_type == "eval" \
                   else train_is_done(j["output"])
            mark = "✓" if done else "○"
            print(f"  {mark}  {j['output']}")
        return

    # Set up environment
    env_overrides = cfg.get("env", {})
    env = {**os.environ, **{k: str(v) for k, v in env_overrides.items()}}

    # Create output & log dirs
    if exp_type == "eval":
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    pid_of: dict[int, subprocess.Popen | None] = {g: None for g in gpus}

    for job in jobs:
        out    = job["output"]
        params = job["params"]

        # Resumability
        if exp_type == "eval" and eval_is_done(out):
            print(f"[skip] {out.name}")
            continue
        if exp_type == "train" and train_is_done(out):
            print(f"[skip] {out} (checkpoint exists)")
            continue

        # Build command
        cli_args = params_to_args(params)
        if exp_type == "eval":
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [python, f"{script}.py"] + cli_args + ["--output", str(out)]
        else:
            out.mkdir(parents=True, exist_ok=True)
            resume_arg = ["--resume", str(out / "checkpoint-last.pth")] \
                         if (out / "checkpoint-last.pth").exists() else []
            cmd = [python, f"{script}.py"] + cli_args + \
                  ["--output_dir", str(out)] + resume_arg

        # Wait for a free GPU
        gpu = None
        while gpu is None:
            for g in gpus:
                p = pid_of[g]
                if p is None or p.poll() is not None:
                    gpu = g
                    break
            if gpu is None:
                time.sleep(20)

        log = log_root / f"{out.stem}.log"
        job_env = {**env, "CUDA_VISIBLE_DEVICES": str(gpu)}
        print(f"[GPU{gpu}] START  {out.name}  ({_ts()})")
        proc = subprocess.Popen(
            cmd, env=job_env,
            stdout=open(log, "w"), stderr=subprocess.STDOUT,
        )
        pid_of[gpu] = proc

    # Drain remaining processes
    for g, p in pid_of.items():
        if p is not None:
            p.wait()
            rc = p.returncode
            if rc != 0:
                print(f"[GPU{g}] WARNING: last job exited with code {rc}")

    print(f"[{_ts()}] ALL DONE — {cfg['name']}")


def _ts() -> str:
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Universal YAML-driven experiment runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("config", help="Path to a YAML experiment config file.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print jobs without running them.")
    ap.add_argument("--gpus", type=int, nargs="+", default=None,
                    help="GPU indices to use (overrides config).")
    args = ap.parse_args()

    cfg  = load_config(args.config)
    gpus = args.gpus if args.gpus is not None else cfg.get("gpus", [0])

    run_all(cfg, gpus, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
