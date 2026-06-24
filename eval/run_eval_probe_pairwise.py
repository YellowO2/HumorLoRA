# ═══════════════════════════════════════════════════════════════════════════════
# Pairwise Crowd Probe: learn crowd preference direction directly from subtask-2
#
# KEY IDEA:
#   Instead of training on binary funny/unfunny labels (old probe), train on
#   crowd pairwise preference labels from subtask-2 train.csv directly.
#
#   For each pair (A, B) where the crowd said A is funnier:
#     - Extract Qwen hidden states h_A and h_B
#     - Compute difference vector h_A - h_B  (3072 numbers)
#     - This is the training example with label = 1 (A preferred)
#
#   Logistic regression on these difference vectors finds direction w_crowd
#   such that w_crowd · (h_A - h_B) > 0 means A is funnier.
#
#   Why better than old probe:
#     - Old probe: 1756 binary funny/unfunny examples → indirect signal
#     - This probe: 1500 crowd pairwise examples → direct supervision
#     - Both headlines are edits of the same headline, so h_A - h_B captures
#       only the humor of the word swap, not background topic/style noise.
#
# Run: python eval/run_eval_probe_pairwise.py
# ═══════════════════════════════════════════════════════════════════════════════

import re
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

PROBE_LAYERS  = [8, 12, 16, 20, 24]
BASE_MODEL    = "unsloth/Qwen3.5-4B"
MAX_LENGTH    = 256
BATCH_SIZE    = 8
SEED          = 42
N_TRAIN       = 3000
N_TEST        = 1000
MIN_MARGIN    = 1.0   # only keep pairs where |meanGrade_A - meanGrade_B| >= this

S2_TRAIN_PATH = Path(__file__).parent.parent / "datasets" / "humicroedit" / "semeval-2020-task-7-dataset" / "subtask-2" / "train.csv"
S2_TEST_PATH  = Path(__file__).parent.parent / "datasets" / "humicroedit" / "semeval-2020-task-7-dataset" / "subtask-2" / "test.csv"
RESULTS_DIR   = Path(__file__).parent.parent / "results" / "probe"
CACHE_DIR     = Path(__file__).parent.parent / "results" / "probe" / "cache"
SUMMARY_PATH  = Path(__file__).parent.parent / "results" / "summary.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROBES_CACHE      = CACHE_DIR / f"humicro_pairwise_probes_margin{MIN_MARGIN}.joblib"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print("Loading Qwen in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", output_hidden_states=True,
    )
    mdl.eval()
    print(f"Model ready. Hidden size: {mdl.config.hidden_size}\n")
    return mdl, tok


def make_prompt(text: str, tok) -> str:
    messages = [{"role": "user", "content": f"Consider the amount of funniness in the following: {text}"}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


@torch.inference_mode()
def get_hidden_states(texts: list[str], layers: list[int], mdl, tok) -> dict[int, np.ndarray]:
    layer_acts = {l: [] for l in layers}
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        prompts = [make_prompt(t, tok) for t in batch]
        enc = tok(prompts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(mdl.device)
        out = mdl(**enc, output_hidden_states=True)
        for l in layers:
            h = out.hidden_states[l][:, -1, :]
            layer_acts[l].append(h.float().cpu().numpy())
        if start % (BATCH_SIZE * 10) == 0:
            print(f"  {min(start + BATCH_SIZE, len(texts))}/{len(texts)} texts")
    return {l: np.concatenate(layer_acts[l], axis=0) for l in layers}


def apply_edit(original: str, edit: str) -> str:
    return re.sub(r"<[^/]+/>", edit, original).strip()


def load_pairs(path: Path, n: int, min_margin: float = 0.0) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["label"] != 0].reset_index(drop=True)
    if min_margin > 0:
        margin = (df["meanGrade1"] - df["meanGrade2"]).abs()
        df = df[margin >= min_margin].reset_index(drop=True)
    return df.iloc[:n]


def ensure_model(mdl, tok):
    if mdl is None:
        mdl, tok = load_model_and_tokenizer()
    return mdl, tok


mdl, tok = None, None

# ── Load pairs ────────────────────────────────────────────────────────────────

print(f"Loading pairs (margin >= {MIN_MARGIN})...")
train_df = load_pairs(S2_TRAIN_PATH, N_TRAIN, MIN_MARGIN)
test_df  = load_pairs(S2_TEST_PATH,  N_TEST,  MIN_MARGIN)
print(f"  Train: {len(train_df)} pairs  |  Test: {len(test_df)} pairs")

train_a = [apply_edit(r.original1, r.edit1) for r in train_df.itertuples()]
train_b = [apply_edit(r.original2, r.edit2) for r in train_df.itertuples()]
test_a  = [apply_edit(r.original1, r.edit1) for r in test_df.itertuples()]
test_b  = [apply_edit(r.original2, r.edit2) for r in test_df.itertuples()]

# labels: 1 = A is funnier, 2 = B is funnier
train_labels = train_df["label"].values
test_labels  = test_df["label"].values


# ── Extract activations ───────────────────────────────────────────────────────

mdl, tok = ensure_model(mdl, tok)

print(f"\nExtracting train activations ({len(train_a) + len(train_b)} texts)...")
raw = get_hidden_states(train_a + train_b, PROBE_LAYERS, mdl, tok)
n = len(train_a)
train_acts = {l: raw[l][:n] - raw[l][n:] for l in PROBE_LAYERS}  # h_A - h_B

print(f"\nExtracting test activations ({len(test_a) + len(test_b)} texts)...")
raw = get_hidden_states(test_a + test_b, PROBE_LAYERS, mdl, tok)
n = len(test_a)
test_acts = {l: raw[l][:n] - raw[l][n:] for l in PROBE_LAYERS}  # h_A - h_B


# ── Fit or load pairwise probes ───────────────────────────────────────────────

# Convert labels: 1 (A funnier) → 1, 2 (B funnier) → 0
y_train = (train_labels == 1).astype(int)
y_test  = (test_labels == 1).astype(int)

if PROBES_CACHE.exists():
    print(f"\nLoading cached probes from {PROBES_CACHE}")
    saved = joblib.load(PROBES_CACHE)
    probes  = saved["probes"]
    scalers = saved["scalers"]
else:
    print("\nFitting pairwise probes per layer...")
    probes, scalers = {}, {}
    for l in PROBE_LAYERS:
        X = train_acts[l]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        clf.fit(X_s, y_train)
        probes[l]  = clf
        scalers[l] = scaler
        print(f"  Layer {l:2d}: done")
    joblib.dump({"probes": probes, "scalers": scalers}, PROBES_CACHE)
    print(f"Probes saved to {PROBES_CACHE}")


# ── Evaluate on test set ──────────────────────────────────────────────────────

print("\n── Test set evaluation ──")
test_accs = {}
for l in PROBE_LAYERS:
    X_s = scalers[l].transform(test_acts[l])
    preds = probes[l].predict(X_s)
    test_accs[l] = (preds == y_test).mean() * 100

best_layer = max(test_accs, key=test_accs.get)

print(f"\n══ Full Results ══")
print(f"{'Layer':>6}  {'Test acc':>9}")
for l in PROBE_LAYERS:
    print(f"  {l:4d}  {test_accs[l]:8.1f}%")

print(f"\nBest layer {best_layer}: {test_accs[best_layer]:.1f}%")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df = pd.DataFrame([
    {"layer": l, "test_acc": round(test_accs[l], 1)}
    for l in PROBE_LAYERS
])
out_path = RESULTS_DIR / f"probe_pairwise_{timestamp}.csv"
results_df.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")

row = {"model": "qwen4b-probe-pairwise", "timestamp": timestamp,
       "n_examples": len(test_df), "overall_acc": round(test_accs[best_layer], 1),
       "unknown_count": 0, "dataset": f"humicro-s2-pairwise-probe-crowd-layer{best_layer}"}
pd.DataFrame([row]).to_csv(SUMMARY_PATH, mode="a", header=not SUMMARY_PATH.exists(), index=False)
print("Summary updated.")
