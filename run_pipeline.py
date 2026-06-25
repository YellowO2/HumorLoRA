"""
Full training pipeline — run this once on the 4090.

Stages:
  1. Prepare: generate datasets/lora_train_data/*.csv from raw sources
  2. Individual: train one LoRA per dataset, eval on HaHa pairwise
  3. Compatibility: cosine similarity between LoRA weight vectors — print report
  4. Joint: train joint LoRA on all compatible datasets (sim > COMPAT_THRESHOLD)

Usage: python run_pipeline.py
"""
import subprocess
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

ROOT           = Path(__file__).parent
LORA_DATA_DIR  = ROOT / "datasets" / "lora_train_data"
CACHE_DIR      = ROOT / "results" / "probe" / "cache"
COMPAT_THRESHOLD = 0.3   # cosine similarity below this → dataset flagged as outlier

DATASETS = ["hahackathon", "humicroedit", "reddit_jokes", "haha_spanish", "humor_arena", "nycc"]

# Rough per-dataset training time estimate on a 4090 (3 epochs, batch=8)
APPROX_MINUTES = {
    "hahackathon":  20,
    "humicroedit":  90,
    "reddit_jokes": 30,
    "haha_spanish": 120,
    "humor_arena":  10,
    "nycc":         25,
}

PYTHON = sys.executable


def run(cmd, label=""):
    print(f"\n{'─'*60}")
    if label:
        print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, check=True)
    return result


# ── Stage 1: Prepare ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("STAGE 1: PREPARE DATA")
print("="*60)

prepare_scripts = {
    "hahackathon":  ROOT / "prepare" / "hahackathon.py",
    "humicroedit":  ROOT / "prepare" / "humicroedit.py",
    "reddit_jokes": ROOT / "prepare" / "reddit_jokes.py",
    "haha_spanish": ROOT / "prepare" / "haha_spanish.py",
    "humor_arena":  ROOT / "prepare" / "humor_arena.py",
    "nycc":         ROOT / "prepare" / "nycc_lora.py",
}

for name, script in prepare_scripts.items():
    out = LORA_DATA_DIR / f"{name}.csv"
    if out.exists():
        print(f"  SKIP {name}: {out} already exists")
    else:
        run([PYTHON, str(script)], f"Prepare {name}")


# ── Stage 2: Individual LoRA training ─────────────────────────────────────────

print("\n" + "="*60)
print("STAGE 2: INDIVIDUAL LORA TRAINING")
print("="*60)

already_done = [n for n in DATASETS if (CACHE_DIR / f"lora_{n}" / "adapter_config.json").exists()]
to_train     = [n for n in DATASETS if n not in already_done]
total_est    = sum(APPROX_MINUTES[n] for n in to_train)
print(f"  To train: {to_train}")
print(f"  Already done (will skip): {already_done}")
print(f"  Estimated time: ~{total_est} min ({total_est//60}h {total_est%60}m)")

for name in DATASETS:
    save = CACHE_DIR / f"lora_{name}"
    if (save / "adapter_config.json").exists():
        print(f"  SKIP {name}: already trained at {save}")
    else:
        run([PYTHON, str(ROOT / "train" / "train_single.py"), "--dataset", name],
            f"Train LoRA: {name}")


# ── Stage 3: Cosine similarity compatibility check ────────────────────────────

print("\n" + "="*60)
print("STAGE 3: COMPATIBILITY CHECK (cosine similarity)")
print("="*60)

def load_lora_vector(name: str) -> np.ndarray:
    """Flatten all LoRA delta weights (B @ A) into one vector."""
    save = CACHE_DIR / f"lora_{name}"

    # Try safetensors first, fall back to pytorch bin
    try:
        from safetensors import safe_open
        weights = {}
        with safe_open(save / "adapter_model.safetensors", framework="pt") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    except (FileNotFoundError, Exception):
        weights = torch.load(save / "adapter_model.bin", map_location="cpu", weights_only=True)

    # Group lora_A / lora_B pairs by module
    modules = {}
    for key, val in weights.items():
        if "lora_A" in key:
            base = key.replace(".lora_A.weight", "")
            modules.setdefault(base, {})["A"] = val.float()
        elif "lora_B" in key:
            base = key.replace(".lora_B.weight", "")
            modules.setdefault(base, {})["B"] = val.float()

    deltas = []
    for base, mats in modules.items():
        if "A" in mats and "B" in mats:
            delta = (mats["B"] @ mats["A"]).flatten()
            deltas.append(delta)

    vec = torch.cat(deltas).numpy()
    return vec / (np.linalg.norm(vec) + 1e-8)


vectors = {}
for name in DATASETS:
    save = CACHE_DIR / f"lora_{name}"
    if not (save / "adapter_config.json").exists():
        print(f"  SKIP {name}: not trained yet")
        continue
    try:
        vectors[name] = load_lora_vector(name)
        print(f"  Loaded LoRA vector: {name}  (dim={vectors[name].shape[0]:,})")
    except Exception as e:
        print(f"  ERROR loading {name}: {e}")

# Pairwise cosine similarity matrix
names = list(vectors.keys())
print(f"\nCosine similarity matrix:")
header = f"{'':15}" + "".join(f"{n:15}" for n in names)
print(header)
sim_matrix = {}
for n1 in names:
    row = f"{n1:15}"
    for n2 in names:
        sim = float(np.dot(vectors[n1], vectors[n2]))
        sim_matrix[(n1, n2)] = sim
        row += f"{sim:+.3f}        "
    print(row)

# Mean similarity of each dataset to all others (excluding self)
print("\nMean similarity to others:")
mean_sims = {}
for n in names:
    others = [sim_matrix[(n, m)] for m in names if m != n]
    mean_sims[n] = np.mean(others)
    flag = "  <-- OUTLIER" if mean_sims[n] < COMPAT_THRESHOLD else ""
    print(f"  {n:20} {mean_sims[n]:+.3f}{flag}")

compatible = [n for n in names if mean_sims[n] >= COMPAT_THRESHOLD]
excluded   = [n for n in names if mean_sims[n] < COMPAT_THRESHOLD]

print(f"\nCompatible datasets (sim >= {COMPAT_THRESHOLD}): {compatible}")
if excluded:
    print(f"Excluded datasets:                             {excluded}")


print("\n" + "="*60)
print("STAGES 1–3 COMPLETE")
print("Review the compatibility report above, then run stage 4 manually:")
print(f"  python train/joint_pairwise.py --datasets {' '.join(compatible)}")
print("="*60)


# ── Stage 4: Joint training (run separately after reviewing stage 3) ──────────
# Uncomment and run once you're happy with the compatibility report.
#
# run(
#     [PYTHON, str(ROOT / "train" / "joint_pairwise.py"), "--datasets", *compatible],
#     "Joint LoRA training",
# )
