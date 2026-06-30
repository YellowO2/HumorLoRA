# HumorLoRA

Code and experiments for **"Can LLMs Learn a Sense of Humor? Crowd-Supervised Fine-Tuning for Humor Preference Judgment"** (URECA@NTU 2025-26).

**Demo:** https://huggingface.co/spaces/potato-bug/humor-judge

## Overview

We study whether LLMs can learn to judge humor as a human crowd would. Prompting-based approaches fail to improve accuracy regardless of framing, CoT, or model size (50--57%). Crowd-supervised LoRA fine-tuning reliably works, reaching ~65--68% and matching average human performance on NYCC.

## Structure

```
prepare/    data preparation scripts (convert datasets to pairwise format)
train/      LoRA training scripts (single-dataset and joint)
eval/       evaluation scripts
results/    experiment results and summary.csv
demo/       HuggingFace Space app
paper/      paper.tex and paper.md
```

## Requirements

```bash
pip install -r requirements.txt
```

Also requires [Ollama](https://ollama.com) for some eval scripts.
