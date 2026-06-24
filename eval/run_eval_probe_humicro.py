# ═══════════════════════════════════════════════════════════════════════════════
# Activation Probe: finding Qwen's internal "funniness direction"
# using Humicroedit (SemEval-2020 Task 7)
# ═══════════════════════════════════════════════════════════════════════════════
#
# KEY IDEA (Representation Engineering / RepE):
#
#   When Qwen processes text, each transformer layer produces a hidden state
#   vector for every token — a point in high-dimensional space (dim ~3072).
#   If Qwen internally "understands" funniness, then funny headlines and
#   unfunny headlines should land in slightly different regions of that space.
#
#   We find this "funniness direction" by:
#     1. Collecting hidden states for ~1000 funny headlines and ~800 unfunny ones
#     2. Fitting a logistic regression: can a linear boundary separate them?
#     3. The weight vector of that classifier IS the funniness direction
#     4. To score a new text: project its hidden state onto that direction
#
#   Caching: activations and probe models are saved to disk so re-runs skip
#   the expensive extraction and fitting steps.
#
# Run: python eval/run_eval_probe_humicro.py
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
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

PROBE_LAYERS  = [8, 12, 16, 20, 24]
BASE_MODEL    = "unsloth/Qwen3.5-4B"
MAX_LENGTH    = 256
BATCH_SIZE    = 8
SEED          = 42

DATA_PATH     = Path(__file__).parent.parent / "datasets" / "humicroedit" / "probe_data.csv"
S2_TEST_PATH  = Path(__file__).parent.parent / "datasets" / "humicroedit" / "semeval-2020-task-7-dataset" / "subtask-2" / "test.csv"
HAHA_PATH     = Path(__file__).parent.parent / "datasets" / "reward" / "test_haha.csv"
RESULTS_DIR   = Path(__file__).parent.parent / "results" / "probe"
CACHE_DIR     = Path(__file__).parent.parent / "results" / "probe" / "cache"
SUMMARY_PATH  = Path(__file__).parent.parent / "results" / "summary.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ACTS_CACHE    = CACHE_DIR / "humicro_probe_acts.npz"
PROBES_CACHE  = CACHE_DIR / "humicro_probes.joblib"


# ── Load model (only if extraction needed) ────────────────────────────────────

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


# ── Step 1: Extract or load probe training activations ────────────────────────

print("Loading Humicroedit probe data...")
df = pd.read_csv(DATA_PATH)
texts  = df["text"].tolist()
labels = df["label"].values
print(f"  Funny: {(labels==1).sum()}  Unfunny: {(labels==0).sum()}")

if ACTS_CACHE.exists():
    print(f"\nLoading cached activations from {ACTS_CACHE}")
    cached = np.load(ACTS_CACHE)
    acts = {int(k): cached[k] for k in cached.files}
else:
    print("\nNo cache found — extracting hidden states (this takes a while)...")
    mdl, tok = load_model_and_tokenizer()
    acts = get_hidden_states(texts, PROBE_LAYERS, mdl, tok)
    np.savez(ACTS_CACHE, **{str(l): acts[l] for l in PROBE_LAYERS})
    print(f"Activations saved to {ACTS_CACHE}")


# ── Step 2: Fit or load probes ────────────────────────────────────────────────

if PROBES_CACHE.exists():
    print(f"\nLoading cached probes from {PROBES_CACHE}")
    saved = joblib.load(PROBES_CACHE)
    probes   = saved["probes"]
    scalers  = saved["scalers"]
    cv_accs  = saved["cv_accs"]
else:
    print("\nFitting linear probes per layer (5-fold CV)...")
    probes, scalers, cv_accs = {}, {}, {}
    for l in PROBE_LAYERS:
        X = acts[l]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        cv_scores = cross_val_score(clf, X_s, labels, cv=5, scoring="accuracy")
        cv_accs[l] = cv_scores.mean() * 100
        clf.fit(X_s, labels)
        probes[l]  = clf
        scalers[l] = scaler
        print(f"  Layer {l:2d}: {cv_accs[l]:.1f}% ± {cv_scores.std()*100:.1f}%")
    joblib.dump({"probes": probes, "scalers": scalers, "cv_accs": cv_accs}, PROBES_CACHE)
    print(f"Probes saved to {PROBES_CACHE}")

best_layer = max(cv_accs, key=cv_accs.get)
print(f"\nBest layer: {best_layer} ({cv_accs[best_layer]:.1f}%)")


# ── Step 3: Score new texts helper ────────────────────────────────────────────

def score_texts(texts: list[str], layer: int, mdl, tok) -> np.ndarray:
    acts_new = get_hidden_states(texts, [layer], mdl, tok)
    X_s = scalers[layer].transform(acts_new[layer])
    return probes[layer].decision_function(X_s)


def ensure_model(mdl, tok):
    if mdl is None:
        mdl, tok = load_model_and_tokenizer()
    return mdl, tok


mdl, tok = None, None


# ── Step 4: In-domain Humicroedit subtask-2 pairwise eval ────────────────────

S2_ACTS_CACHE = CACHE_DIR / "humicro_s2_acts.npz"

print("\n── In-domain eval: Humicroedit subtask-2 pairwise ──")
s2 = pd.read_csv(S2_TEST_PATH)
s2 = s2[s2["label"] != 0].reset_index(drop=True)
print(f"Loaded {len(s2)} non-tie pairs")

headlines_a = [apply_edit(r.original1, r.edit1) for r in s2.itertuples()]
headlines_b = [apply_edit(r.original2, r.edit2) for r in s2.itertuples()]
all_headlines = headlines_a + headlines_b

if S2_ACTS_CACHE.exists():
    print(f"Loading cached subtask-2 activations from {S2_ACTS_CACHE}")
    cached_s2 = np.load(S2_ACTS_CACHE)
    s2_acts = {int(k): cached_s2[k] for k in cached_s2.files}
else:
    mdl, tok = ensure_model(mdl, tok)
    print(f"Extracting hidden states for {len(all_headlines)} subtask-2 headlines...")
    s2_acts = get_hidden_states(all_headlines, PROBE_LAYERS, mdl, tok)
    np.savez(S2_ACTS_CACHE, **{str(l): s2_acts[l] for l in PROBE_LAYERS})
    print(f"Subtask-2 activations saved to {S2_ACTS_CACHE}")

n = len(s2)
s2_accs = {}
for l in PROBE_LAYERS:
    X_s = scalers[l].transform(s2_acts[l])
    scores = probes[l].decision_function(X_s)
    scores_a = scores[:n]
    scores_b = scores[n:]
    predicted = np.where(scores_a > scores_b, 1, 2)
    correct = (predicted == s2["label"].values).mean() * 100
    s2_accs[l] = correct

print(f"\nSubtask-2 pairwise accuracy per layer:")
for l in PROBE_LAYERS:
    print(f"  Layer {l:2d}: {s2_accs[l]:.1f}%")


# ── Step 5: Cross-domain HaHa pairwise eval (commented out for now) ───────────

# HAHA_ACTS_CACHE = CACHE_DIR / "haha_xdomain_acts.npz"
# haha = pd.read_csv(HAHA_PATH)
# haha_texts = haha["chosen"].tolist() + haha["rejected"].tolist()
# if HAHA_ACTS_CACHE.exists():
#     cached_haha = np.load(HAHA_ACTS_CACHE)
#     haha_acts = {int(k): cached_haha[k] for k in cached_haha.files}
# else:
#     mdl, tok = ensure_model(mdl, tok)
#     haha_acts = get_hidden_states(haha_texts, PROBE_LAYERS, mdl, tok)
#     np.savez(HAHA_ACTS_CACHE, **{str(l): haha_acts[l] for l in PROBE_LAYERS})
# nh = len(haha)
# haha_accs = {}
# for l in PROBE_LAYERS:
#     X_s = scalers[l].transform(haha_acts[l])
#     scores = probes[l].decision_function(X_s)
#     acc = (scores[:nh] > scores[nh:]).mean() * 100
#     haha_accs[l] = acc
haha_accs = {}  # empty — HaHa eval skipped


# ── Results ───────────────────────────────────────────────────────────────────

print("\n══ Full Results ══")
print(f"{'Layer':>6}  {'Humicro CV':>11}  {'Humicro S2':>11}")
for l in PROBE_LAYERS:
    print(f"  {l:4d}  {cv_accs[l]:10.1f}%  {s2_accs[l]:10.1f}%")

print(f"\nBest layer {best_layer}:")
print(f"  Humicro binary CV:     {cv_accs[best_layer]:.1f}%")
print(f"  Humicro S2 pairwise:   {s2_accs[best_layer]:.1f}%")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df = pd.DataFrame([
    {"layer": l, "humicro_cv": round(cv_accs[l], 1),
     "humicro_s2_pairwise": round(s2_accs[l], 1)}
    for l in PROBE_LAYERS
])
out_path = RESULTS_DIR / f"probe_humicro_{timestamp}.csv"
results_df.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")

row = {"model": "qwen4b-probe-humicro", "timestamp": timestamp,
       "n_examples": len(s2), "overall_acc": round(s2_accs[best_layer], 1),
       "unknown_count": 0, "dataset": f"humicro-s2-pairwise-probe-layer{best_layer}"}
pd.DataFrame([row]).to_csv(SUMMARY_PATH, mode="a", header=not SUMMARY_PATH.exists(), index=False)
print("Summary updated.")
