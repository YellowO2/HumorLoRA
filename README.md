# Fine-tuning for Human Alignment

Experiment: does fine-tuning small models on Discord conversations improve
their ability to judge human humor preferences? Evaluated on the New Yorker
Caption Contest (NYCC) — given a cartoon and two captions, pick the funnier one.


## Goal

Take base models, fine-tune them on Discord chat data (SFT), and see if they
get better at predicting which caption humans found funnier. Baseline is each
model's accuracy before any fine-tuning.


## Models

- gemma4:e2b   — Gemma 4 E2B (~2B), fine-tuned -> gemma4-e2b-discord
- gemma4:e4b   — Gemma 4 E4B (~4B), fine-tuned -> gemma4-e4b-discord
- qwen3.5:9b   — Qwen 3.5 9B, fine-tuned -> qwen9b-discord
- qwen3:4b     — Qwen 3 4B (baseline only so far)


## Results so far

Accuracy on NYCC humor preference (random = 50%):

  model                 examples   accuracy
  --------------------  ---------  --------
  gemma4:e2b   (base)      2600      52.3%
  gemma4:e4b   (base)      2600      53.7%
  qwen3.5:9b   (base)      2600      52.4%
  qwen3:4b     (base)       300      52.0%   <- old (Qwen 3, not 3.5), only 300 examples
  qwen3.5:4b   (base)      2616      55.9%   <- best baseline
  gemma4-e2b-discord       2616      49.8%   <- fine-tuned, WORSE
  gemma4-e4b-discord       2616      50.6%   <- fine-tuned, WORSE

Key finding: Discord SFT made both Gemma models WORSE (dropped to ~chance).
Likely cause is task mismatch — we trained on chat *generation* but evaluate
on preference *discrimination*. The models learned to chat and lost their
preference-judging ability (symptom: heavy position bias toward "A").

qwen3.5:4b is the strongest baseline at 55.9% — good DPO starting point.


## Architecture gotcha (Gemma 4 E2B / E4B)

Gemma 4 E2B and E4B use KV-shared layers: ~20 transformer layers share the
same K and V weight tensors in memory. Consequences:

- LoRA inference WITHOUT merging works fine (each layer adds its own
  correction at runtime).
- LoRA MERGE is broken — you'd need to fold 20 different corrections into one
  shared tensor, which corrupts the weights. Produces <unused> garbage tokens.
- So GGUF export / Ollama does NOT work for fine-tuned Gemma 4 E2B/E4B.
- We serve these checkpoints directly via unsloth FastLanguageModel instead
  (see eval/interact.py LocalModel).

Qwen has standard architecture — no KV sharing — so its LoRA merges and GGUF
export work normally (Ollama-compatible). qwen9b-discord is exported and loaded
into Ollama as `qwen9b-discord`.

Note on the export: unsloth's `save_pretrained_gguf` fails on current llama.cpp
("No module named 'conversion'") — it patches the convert script into a temp dir
and loses its package context. That is NOT an install problem (training and
merging work fine). The fix is to do the export manually with the real local
llama.cpp script — see training/export_qwen_gguf.py:
  1. merge LoRA -> 16-bit HF model       (save_pretrained_merged)
  2. HF -> GGUF f16                       (~/.unsloth/llama.cpp/convert_hf_to_gguf.py)
  3. quantize f16 -> q4_k_m               (~/.unsloth/llama.cpp/llama-quantize)
  4. ollama create -f Modelfile


## Files

  training/
    train_sft.py           Gemma 4 E2B SFT (original)
    train_sft_e4b.py       Gemma 4 E4B SFT
    train_sft_qwen9b.py    Qwen 3.5 9B SFT
    export_qwen_gguf.py    Qwen LoRA -> merged -> GGUF -> q4_k_m (works)
    export_gguf.py         standalone GGUF export attempt (Gemma, abandoned)

  eval/
    interact.py            LocalModel (FastLanguageModel inference) + Ollama ask()
    run_eval.py            runs NYCC eval over MODELS, writes results/

  chat.py                  terminal REPL to chat with a fine-tuned checkpoint
                           usage: python chat.py e2b | e4b | <checkpoint path>

  outputs/                 LoRA checkpoints (gemma4-e2b/e4b-discord, qwen9b-discord)
  results/                 per-run CSVs + summary.csv
  datasets/
    discord/sft.jsonl      Discord training data
    newyorker/             NYCC eval folds


## How to run

Train:
  python training/train_sft_e4b.py
  python training/train_sft_qwen9b.py

Eval (make sure Ollama has nothing loaded — `ollama ps` — to free VRAM):
  python eval/run_eval.py

Chat:
  python chat.py e4b               # Gemma checkpoints, via FastLanguageModel
  ollama run qwen9b-discord        # Qwen, via Ollama (shareable GGUF)

Export Qwen to GGUF / Ollama:
  python training/export_qwen_gguf.py all
  ollama create qwen9b-discord -f outputs/qwen9b-discord-gguf/Modelfile


## Next steps
- SFT is the wrong tool — pivoting to DPO.

### DPO direction

Hypothesis: training the model to prefer human-style responses over AI-style
responses (human alignment) will nudge its internal "taste" closer to human
judgment, which should transfer to unseen human preference tasks like NYCC
humor selection.

Key principle: train and test on DIFFERENT datasets (cross-dataset transfer).
Training on NYCC directly would just be memorization, not alignment.

Training script: training/train_dpo_qwen9b_degpt.py
Dataset: qingy2024/De-GPT-DPO (44.7k pairs)
  chosen   = real human-written Reddit response (short, casual, direct)
  rejected = AI/GPT-style response (verbose, structured, formal)

Eval: run_eval.py on NYCC after DPO — does human-alignment transfer to
humor preference judgment?

