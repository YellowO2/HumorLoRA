# Datasets

We assemble five crowd-labeled humor datasets spanning different domains, label types, and languages, and unify them into a single pairwise training format.

## 4.1 Overview

| Dataset | Domain | Label type | Train rows | Source |
|---|---|---|---|---|
| hahackathon | Short text jokes | Crowd rating 1–5 | ~3.9k | Meaney et al. (2021) |
| humicroedit | News headline edits | Crowd rating 0–3 | ~10k (capped) | Hossain et al. (2020) |
| reddit\_jokes | Reddit posts | Upvote ratio | ~10k (capped) | Weller & Seppi (2019) |
| haha\_spanish | Spanish text jokes | Crowd rating 1–5 | ~10k (capped) | Chiruzzo et al. (2019) |
| nycc | New Yorker cartoon captions | Native pairwise | 2k pairs | Hessel et al. (2023) |

## 4.2 Data Preparation

Rated datasets (hahackathon, humicroedit, haha\_spanish) are converted to pairwise format by sampling (high-score, low-score) pairs within each source. reddit\_jokes uses upvote ratio as a funniness proxy with a score ≥ 10 filter and deduplication. nycc is already native pairwise; we use fold0 only with non-overlapping train/test/val splits, and include a text description of the cartoon image since we work in a text-only setting. Pairwise format removes cross-dataset label scale differences and allows a uniform training objective across all five sources.

Each dataset is capped at 10k rows to prevent larger datasets from dominating. hahackathon is used in full (~3.9k). Total after capping is roughly 44k rows, yielding around 22k training pairs. Two benchmarks are held out for evaluation: **HaHa pairwise** (N=2000, 20% split from hahackathon) and **NYCC pairwise** (N=1020, fold0 test+val, human ceiling 64.6%). Humicroedit pairwise (N=2628) is used as an additional eval for probing and single-dataset LoRA experiments.

## 4.3 Considered but Excluded

- **humor\_arena** (~2.5k pairwise, LLM-generated jokes): a LoRA trained on humor\_arena scored 52.8% cross-domain on HaHa pairwise, below the zero-shot baseline (-2.6pp), so it was excluded from joint training.
- **Jester** (dataset 3): only 140 unique jokes (54k raters); not tried. [TODO: could generate ~9,700 pairs from 140 jokes — worth testing if it contributes signal despite low item diversity]
- **Open Mic** (Mittal et al.): stand-up transcripts with audience laughter as ground truth; not publicly accessible.
- **Cards Against Humanity (CAH Lab)**: requires contacting the lab directly; did not respond.
- **Yelp Funny Reviews**: not explored due to time constraints.
- **Oogiri-GO** (Murakami et al.): not explored due to time constraints.
