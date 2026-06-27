"""
Eval the saved joint LoRA on HaHa, NYCC, and Humicroedit pairwise benchmarks.

Usage: python eval/eval_joint_lora.py
"""
import gc
import re
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

BASE_MODEL   = "unsloth/Qwen3.5-4B"
TARGET_LAYER = 16
MAX_LENGTH   = 192
EVAL_BATCH   = 4

ROOT         = Path(__file__).parent.parent
MODEL_SAVE   = ROOT / "results" / "probe" / "cache" / "lora_joint_pairwise"
SUMMARY_PATH = ROOT / "results" / "summary.csv"

_log(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}, cuda memory: {torch.cuda.mem_get_info() if torch.cuda.is_available() else 'N/A'}")

_log("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
_log("Tokenizer loaded")

_log("Loading base model in 4-bit...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto", local_files_only=True,
)
_log("Base model loaded, applying LoRA...")
base = PeftModel.from_pretrained(base, MODEL_SAVE)
base.eval()
_log("LoRA applied")

dev  = next(base.parameters()).device
_log(f"Model device: {dev}, loading head...")
head = nn.Linear(base.config.hidden_size, 1).to(dev)
head.load_state_dict(torch.load(MODEL_SAVE / "head.pt", map_location=dev))
head.eval()

_log(f"Loaded joint LoRA from {MODEL_SAVE}")


def make_prompt(text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


@torch.inference_mode()
def score_all(prompts):
    out = []
    for i in range(0, len(prompts), EVAL_BATCH):
        enc = tok(prompts[i:i+EVAL_BATCH], truncation=True, max_length=MAX_LENGTH,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        h   = base(**enc, output_hidden_states=True).hidden_states[TARGET_LAYER][:, -1, :].to(dev)
        out.extend(head(h.float()).squeeze(-1).cpu().tolist())
        if i % (EVAL_BATCH * 50) == 0:
            print(f"  {min(i+EVAL_BATCH, len(prompts))}/{len(prompts)}")
    return np.array(out)


def log_result(model_name, n, acc, dataset_tag):
    row = {"model": model_name, "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "n_examples": n, "overall_acc": round(acc, 1), "unknown_count": 0,
           "dataset": dataset_tag}
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(SUMMARY_PATH, mode="a", header=not SUMMARY_PATH.exists(), index=False)
    print(f"  Logged to {SUMMARY_PATH}")


# ── HaHa pairwise ─────────────────────────────────────────────────────────────
# haha_path = ROOT / "datasets" / "hahackathon" / "pairwise.csv"
# if haha_path.exists():
#     print("\n── Eval: HaHa pairwise ──")
#     pw      = pd.read_csv(haha_path)
#     prefix  = "Consider the amount of funniness in the following: "
#     prompts = [make_prompt(prefix + j) for j in pw["text_a"].tolist() + pw["text_b"].tolist()]
#     scores  = score_all(prompts)
#     n       = len(pw)
#     acc     = (np.where(scores[:n] > scores[n:], "A", "B") == np.array(pw["expected"].tolist())).mean() * 100
#     print(f"HaHa pairwise accuracy: {acc:.1f}%  (n={n})")
#     log_result("qwen4b-lora-joint", n, acc, "joint-lora-haha")
# 
# 
# # ── NYCC pairwise ─────────────────────────────────────────────────────────────
# nycc_path = ROOT / "datasets" / "nycc_pairwise_test.csv"
# if nycc_path.exists():
#     print("\n── Eval: NYCC pairwise ──")
#     pw = pd.read_csv(nycc_path)
# 
#     def build_nycc_prompt(caption, image_desc):
#         return (
#             f"Consider the amount of funniness in the following New Yorker cartoon caption.\n\n"
#             f"Image: {image_desc.strip()}\n"
#             f"Caption: {caption.strip()}"
#         )
# 
#     prompts_a = [make_prompt(build_nycc_prompt(c, img))
#                  for c, img in zip(pw["caption_a"], pw["image_description"])]
#     prompts_b = [make_prompt(build_nycc_prompt(c, img))
#                  for c, img in zip(pw["caption_b"], pw["image_description"])]
#     scores = score_all(prompts_a + prompts_b)
#     n      = len(pw)
#     acc    = (np.where(scores[:n] > scores[n:], "A", "B") == np.array(pw["expected"].tolist())).mean() * 100
#     print(f"NYCC pairwise accuracy: {acc:.1f}%  (n={n})")
#     log_result("qwen4b-lora-joint", n, acc, "joint-lora-nycc")


# ── Humicroedit pairwise ──────────────────────────────────────────────────────
humicroedit_path = ROOT / "datasets" / "humicroedit" / "semeval-2020-task-7-dataset" / "subtask-2" / "test.csv"
if humicroedit_path.exists():
    _log("Eval: Humicroedit pairwise")

    def apply_edit(original, edit):
        return re.sub(r"<[^/]+/>", edit, original).strip()

    s2 = pd.read_csv(humicroedit_path)
    s2 = s2[s2["label"] != 0].reset_index(drop=True)
    _log(f"{len(s2)} non-tie pairs loaded")

    prefix = "Consider the amount of funniness in the following: "
    headlines_a = [apply_edit(r.original1, r.edit1) for r in s2.itertuples()]
    headlines_b = [apply_edit(r.original2, r.edit2) for r in s2.itertuples()]
    prompts = [make_prompt(prefix + h) for h in headlines_a + headlines_b]
    _log(f"Scoring {len(prompts)} prompts...")
    scores = score_all(prompts)
    n = len(s2)
    predicted = np.where(scores[:n] > scores[n:], 1, 2)
    acc = (predicted == s2["label"].values).mean() * 100
    _log(f"Humicroedit pairwise accuracy: {acc:.1f}%  (n={n})")
    log_result("qwen4b-lora-joint", n, acc, "joint-lora-humicroedit")


gc.collect()
torch.cuda.empty_cache()
_log("Done.")
