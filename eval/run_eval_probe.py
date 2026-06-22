# ═══════════════════════════════════════════════════════════════════════════════
# Activation probe: does Qwen already encode funniness internally?
# ═══════════════════════════════════════════════════════════════════════════════
#
# Method (Representation Engineering / RepE):
#   1. Run "React to this: [joke]" through Qwen (no reward head, base LM only)
#   2. Extract hidden-state activations at the last token, across multiple layers
#   3. Fit a logistic regression probe: chosen=1 vs rejected=0
#   4. Eval probe accuracy on held-out HaHa test + NYCC pairwise
#
# If probe accuracy >> 50%, Qwen already has a linear funniness direction.
# This is a diagnostic before deciding whether to train further.
#
# Run: python eval/run_eval_probe.py
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROBE_LAYERS  = [8, 12, 16, 20, 24]   # which transformer layers to probe
BASE_MODEL    = "unsloth/Qwen3.5-4B"
PROMPT_PREFIX = "React to this: "
MAX_LENGTH    = 256
BATCH_SIZE    = 8
SEED          = 42

TRAIN_PATH    = Path(__file__).parent.parent / "datasets" / "reward" / "train.csv"
TEST_HAHA     = Path(__file__).parent.parent / "datasets" / "reward" / "test_haha.csv"
NYCC_DIR      = Path(__file__).parent.parent / "datasets" / "newyorker"
RESULTS_DIR   = Path(__file__).parent.parent / "results" / "probe"
SUMMARY_PATH  = Path(__file__).parent.parent / "results" / "summary.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Load model (base LM, no classification head) ──────────────────────────────

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # causal LM: last token is what we extract

print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    output_hidden_states=True,
)
model.eval()
print("Model ready.\n")


# ── Extract activations ────────────────────────────────────────────────────────

@torch.inference_mode()
def get_activations(texts: list[str], layers: list[int]) -> dict[int, np.ndarray]:
    """Returns {layer: (N, hidden_size)} array of last-token activations."""
    layer_acts = {l: [] for l in layers}

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        prompts = [PROMPT_PREFIX + t for t in batch]
        enc = tokenizer(
            prompts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        out = model(**enc)
        # hidden_states: tuple of (batch, seq_len, hidden_size) for each layer
        # index 0 = embedding, index 1..N = transformer layers
        for l in layers:
            # last non-padding token = last token (left-padded, so it's always [-1])
            hidden = out.hidden_states[l][:, -1, :]   # (batch, hidden_size)
            layer_acts[l].append(hidden.float().cpu().numpy())

        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  activations: {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")

    return {l: np.concatenate(layer_acts[l], axis=0) for l in layers}


# ── Load training data ─────────────────────────────────────────────────────────

print("Loading training pairs...")
train_df = pd.read_csv(TRAIN_PATH)
# sample to keep memory reasonable — 2000 chosen + 2000 rejected
rng = np.random.default_rng(SEED)
idx = rng.choice(len(train_df), min(2000, len(train_df)), replace=False)
train_df = train_df.iloc[idx].reset_index(drop=True)

train_texts  = train_df["chosen"].tolist() + train_df["rejected"].tolist()
train_labels = [1] * len(train_df) + [0] * len(train_df)

print(f"Extracting activations for {len(train_texts)} training texts...")
train_acts = get_activations(train_texts, PROBE_LAYERS)


# ── Fit probes ────────────────────────────────────────────────────────────────

print("\nFitting probes per layer...")
probes   = {}
scalers  = {}
train_accs = {}

for l in PROBE_LAYERS:
    X = train_acts[l]
    y = np.array(train_labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    clf.fit(X_scaled, y)

    train_acc = clf.score(X_scaled, y) * 100
    probes[l]  = clf
    scalers[l] = scaler
    train_accs[l] = train_acc
    print(f"  Layer {l:2d}: train acc = {train_acc:.1f}%")


# ── Eval helper ───────────────────────────────────────────────────────────────

def eval_pairwise(texts_a: list[str], texts_b: list[str], labels: list[int], layer: int) -> float:
    """Returns pairwise accuracy: fraction where score(a) > score(b) matches label."""
    all_texts = texts_a + texts_b
    acts = get_activations(all_texts, [layer])
    X = acts[layer]
    X_scaled = scalers[layer].transform(X)
    scores = probes[layer].decision_function(X_scaled)   # higher = more likely chosen

    n = len(texts_a)
    scores_a = scores[:n]
    scores_b = scores[n:]
    predictions = (scores_a > scores_b).astype(int)
    return float((predictions == np.array(labels)).mean())


# ── Eval: held-out HaHa ───────────────────────────────────────────────────────

print("\n── Eval: held-out HaHa ──")
test_haha = pd.read_csv(TEST_HAHA)
texts_chosen   = test_haha["chosen"].tolist()
texts_rejected = test_haha["rejected"].tolist()
labels_haha    = [1] * len(test_haha)   # chosen is always funnier

haha_accs = {}
for l in PROBE_LAYERS:
    acc = eval_pairwise(texts_chosen, texts_rejected, labels_haha, l)
    haha_accs[l] = acc * 100
    print(f"  Layer {l:2d}: {acc*100:.1f}%")


# ── Eval: NYCC ────────────────────────────────────────────────────────────────

print("\n── Eval: NYCC ──")
dfs = []
for fold in range(5):
    df = pd.read_csv(NYCC_DIR / f"fold{fold}_validation.csv")
    df["fold"] = fold
    dfs.append(df)
nycc = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(nycc)} NYCC pairs")

texts_a = [f"Image: {r.image_description}\n{r.caption_a}" for r in nycc.itertuples()]
texts_b = [f"Image: {r.image_description}\n{r.caption_b}" for r in nycc.itertuples()]
labels_nycc = [1 if str(r.expected).strip().upper() == "A" else 0 for r in nycc.itertuples()]

nycc_accs = {}
for l in PROBE_LAYERS:
    acc = eval_pairwise(texts_a, texts_b, labels_nycc, l)
    nycc_accs[l] = acc * 100
    print(f"  Layer {l:2d}: {acc*100:.1f}%")


# ── Summary ───────────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print("\n══ Results ══")
print(f"{'Layer':>6}  {'Train':>7}  {'HaHa':>7}  {'NYCC':>7}")
for l in PROBE_LAYERS:
    print(f"  {l:4d}  {train_accs[l]:6.1f}%  {haha_accs[l]:6.1f}%  {nycc_accs[l]:6.1f}%")

# pick best layer by HaHa accuracy
best_layer = max(haha_accs, key=haha_accs.get)
print(f"\nBest layer (by HaHa): {best_layer}  →  HaHa {haha_accs[best_layer]:.1f}%  NYCC {nycc_accs[best_layer]:.1f}%")

results_df = pd.DataFrame([
    {
        "layer": l,
        "train_acc": round(train_accs[l], 1),
        "haha_acc":  round(haha_accs[l], 1),
        "nycc_acc":  round(nycc_accs[l], 1),
    }
    for l in PROBE_LAYERS
])
results_df.to_csv(RESULTS_DIR / f"probe_{timestamp}.csv", index=False)
print(f"Results saved to {RESULTS_DIR / f'probe_{timestamp}.csv'}")

row = {
    "model":       "qwen4b-probe",
    "timestamp":   timestamp,
    "n_examples":  len(nycc),
    "overall_acc": round(nycc_accs[best_layer], 1),
    "unknown_count": 0,
    "dataset":     f"nycc-probe-layer{best_layer}",
}
summary_df = pd.DataFrame([row])
if SUMMARY_PATH.exists():
    summary_df.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)
else:
    summary_df.to_csv(SUMMARY_PATH, index=False)
print("Summary updated.")
