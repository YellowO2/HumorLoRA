"""
Train a single LoRA on one dataset using in-batch pairwise BCE loss.
Called by run_pipeline.py for each dataset in the compatibility check stage.

Usage: python train/train_single.py --dataset hahackathon
Saves: results/probe/cache/lora_{dataset}/
"""
import argparse
import gc
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "unsloth/Qwen3.5-4B"
TARGET_LAYER = 16
LORA_R       = 16
LORA_ALPHA   = 32
LR           = 2e-4
EPOCHS       = 3
BATCH_SIZE   = 8
GRAD_ACCUM   = 2
MAX_LENGTH   = 192
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent.parent
LORA_DATA_DIR = ROOT / "datasets" / "lora_train_data"
CACHE_DIR     = ROOT / "results" / "probe" / "cache"
SUMMARY_PATH  = ROOT / "results" / "summary.csv"

torch.manual_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (must exist in datasets/lora_train_data/)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    return parser.parse_args()


def make_prompt(tok, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


class JokeDataset(Dataset):
    def __init__(self, prompts, scores, tok):
        self.prompts = [make_prompt(tok, p) for p in prompts]
        self.scores  = scores
        self.tok     = tok

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.prompts[idx], truncation=True, max_length=MAX_LENGTH,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "score":          torch.tensor(self.scores[idx], dtype=torch.float32),
        }


def pairwise_bce_loss(pred_scores, true_scores):
    N = pred_scores.size(0)
    diff_pred = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)
    diff_true = true_scores.unsqueeze(1) - true_scores.unsqueeze(0)
    mask = (torch.triu(torch.ones(N, N, device=pred_scores.device), diagonal=1) == 1) \
           & (diff_true != 0)
    if mask.sum() == 0:
        return pred_scores.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(diff_pred[mask], (diff_true[mask] > 0).float())


def main():
    args  = parse_args()
    name  = args.dataset
    epochs = args.epochs
    dpath = LORA_DATA_DIR / f"{name}.csv"
    save  = CACHE_DIR / f"lora_{name}"
    save.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training LoRA: {name}")
    print(f"{'='*60}")

    df = pd.read_csv(dpath)
    print(f"Loaded {len(df)} rows from {dpath}")

    print("Loading tokenizer...")
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dataset = JokeDataset(df["prompt_text"].tolist(), df["score"].tolist(), tok)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    print("Loading model in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    try:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
    except Exception:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", local_files_only=True)
    base = prepare_model_for_kbit_training(base)
    base = get_peft_model(base, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    ))
    base.print_trainable_parameters()

    dev  = next(base.parameters()).device
    head = nn.Linear(base.config.hidden_size, 1).to(dev)

    optimizer   = torch.optim.AdamW(list(base.parameters()) + list(head.parameters()), lr=LR, weight_decay=0.01)
    total_steps = (len(loader) // GRAD_ACCUM) * epochs
    scheduler   = get_linear_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)

    total_steps = len(loader) * epochs
    print(f"Training {epochs} epochs on {len(dataset)} examples  ({total_steps} steps total)...")
    pipeline_start = time.time()

    for epoch in range(epochs):
        base.train(); head.train()
        total_loss = 0.0
        optimizer.zero_grad()
        epoch_start = time.time()

        for step, batch in enumerate(loader):
            ids  = batch["input_ids"].to(dev)
            mask = batch["attention_mask"].to(dev)
            true = batch["score"].to(dev)
            out  = base(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            h    = out.hidden_states[TARGET_LAYER][:, -1, :].to(dev)
            pred = head(h.float()).squeeze(-1)
            loss = pairwise_bce_loss(pred, true) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(list(base.parameters()) + list(head.parameters()), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if (step + 1) % 50 == 0:
                elapsed   = time.time() - pipeline_start
                done_steps = epoch * len(loader) + step + 1
                sps        = done_steps / elapsed
                remaining  = (total_steps - done_steps) / sps
                eta_min    = int(remaining // 60)
                eta_sec    = int(remaining % 60)
                print(f"  Epoch {epoch+1} step {step+1}/{len(loader)}  "
                      f"loss={total_loss/(step+1):.4f}  "
                      f"ETA {eta_min}m{eta_sec:02d}s")

        epoch_elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} done. Avg loss: {total_loss/len(loader):.4f}  "
              f"({epoch_elapsed/60:.1f} min)")

    base.save_pretrained(save)
    torch.save(head.state_dict(), save / "head.pt")
    print(f"Saved to {save}")

    # ── Shared scorer ─────────────────────────────────────────────────────────
    base.eval(); head.eval()

    @torch.inference_mode()
    def score_all(prompts):
        out = []
        for i in range(0, len(prompts), BATCH_SIZE):
            enc = tok(prompts[i:i+BATCH_SIZE], truncation=True, max_length=MAX_LENGTH,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            h   = base(**enc, output_hidden_states=True).hidden_states[TARGET_LAYER][:, -1, :].to(dev)
            out.extend(head(h.float()).squeeze(-1).cpu().tolist())
        return np.array(out)

    def log_result(model_name, n, acc, dataset_tag):
        row = {"model": model_name, "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
               "n_examples": n, "overall_acc": round(acc, 1), "unknown_count": 0,
               "dataset": dataset_tag}
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(SUMMARY_PATH, mode="a", header=not SUMMARY_PATH.exists(), index=False)

    # ── Eval on HaHa pairwise benchmark ──────────────────────────────────────
    pairwise_path = ROOT / "datasets" / "hahackathon" / "pairwise.csv"
    if pairwise_path.exists():
        print("\nEval on HaHa pairwise held-out test...")
        pw      = pd.read_csv(pairwise_path)
        prefix  = "Consider the amount of funniness in the following: "
        prompts = [make_prompt(tok, prefix + j) for j in pw["text_a"].tolist() + pw["text_b"].tolist()]
        scores  = score_all(prompts)
        n       = len(pw)
        acc     = (np.where(scores[:n] > scores[n:], "A", "B") == np.array(pw["expected"].tolist())).mean() * 100
        print(f"HaHa pairwise accuracy ({name}): {acc:.1f}%")
        log_result(f"qwen4b-lora-{name}", n, acc, f"individual-lora-layer{TARGET_LAYER}")

    # ── Eval on NYCC pairwise benchmark (when training on NYCC) ──────────────
    nycc_test_path = ROOT / "datasets" / "nycc_pairwise_test.csv"
    if name == "nycc" and nycc_test_path.exists():
        print("\nEval on NYCC pairwise held-out test...")
        pw = pd.read_csv(nycc_test_path)

        def build_nycc_prompt(caption: str, image_desc: str) -> str:
            return (
                f"Consider the amount of funniness in the following New Yorker cartoon caption.\n\n"
                f"Image: {image_desc.strip()}\n"
                f"Caption: {caption.strip()}"
            )

        prompts_a = [make_prompt(tok, build_nycc_prompt(c, img))
                     for c, img in zip(pw["caption_a"], pw["image_description"])]
        prompts_b = [make_prompt(tok, build_nycc_prompt(c, img))
                     for c, img in zip(pw["caption_b"], pw["image_description"])]
        scores = score_all(prompts_a + prompts_b)
        n      = len(pw)
        acc    = (np.where(scores[:n] > scores[n:], "A", "B") == np.array(pw["expected"].tolist())).mean() * 100
        print(f"NYCC pairwise accuracy: {acc:.1f}%  (n={n})")
        log_result(f"qwen4b-lora-{name}", n, acc, f"nycc-lora-layer{TARGET_LAYER}")

    del base, head, optimizer, scheduler, dataset, loader
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Done: {name}")


if __name__ == "__main__":
    main()
