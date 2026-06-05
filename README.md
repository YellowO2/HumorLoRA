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
  qwen3:4b     (base)       300      52.0%
  gemma4-e2b-discord       2616      49.8%   <- fine-tuned, WORSE
  gemma4-e4b-discord       2616      50.6%   <- fine-tuned, WORSE

Key finding: Discord SFT made both Gemma models WORSE (dropped to ~chance).
Likely cause is task mismatch — we trained on chat *generation* but evaluate
on preference *discrimination*. The models learned to chat and lost their
preference-judging ability (symptom: heavy position bias toward "A").


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
export work normally (Ollama-compatible). GGUF export currently blocked by a
separate unsloth/llama.cpp converter bug (No module named 'conversion').


## Files

  training/
    train_sft.py           Gemma 4 E2B SFT (original)
    train_sft_e4b.py       Gemma 4 E4B SFT
    train_sft_qwen9b.py    Qwen 3.5 9B SFT (+ GGUF export, currently failing)
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
  python chat.py e4b


## Next steps

- Qwen 9B: trained, needs eval (GGUF export optional / currently broken).
- The real fix for the accuracy drop is task alignment, not more chat SFT:
    1. Fine-tune directly on the NYCC preference task (A/B labels), OR
    2. Try DPO using NYCC preference pairs (chosen = funnier caption,
       rejected = the other). This matches the eval objective directly and
       is the most promising direction.
