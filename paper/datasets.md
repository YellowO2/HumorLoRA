# Datasets

We assemble five crowd-labeled humor datasets spanning different domains, label types, and languages, and unify them into a single pairwise training format.

## 4.1 Overview

| Dataset | Domain | Label type | Train rows | Source |
|---|---|---|---|---|
| hahackathon | Short text jokes | Crowd rating 1–5 | ~3.9k | Meaney et al. (2021) |
| humicroedit | News headline edits | Crowd rating 0–3 | ~10k (capped) | Hossain et al. (2020) |
| reddit\_jokes | Reddit posts | Upvote ratio | ~10k (capped) | Weller & Seppi (2019) |
| haha\_spanish | Spanish twitter jokes | Crowd rating 1–5 | ~10k (capped) | Chiruzzo et al. (2019) |
| nycc | New Yorker cartoon captions | Native pairwise | 2k pairs | Hessel et al. (2023) |

## 4.2 Data Preparation

Rated datasets (hahackathon, humicroedit, haha\_spanish) are converted to pairwise format by sampling (high-score, low-score) pairs within each source. reddit\_jokes uses upvote ratio as a funniness proxy with a score ≥ 10 filter and deduplication. nycc is already native pairwise; we use fold0 only with non-overlapping train/test/val splits, and use text description of the cartoon image since we work in a text-only setting. 
By having a uniform dataset format in pairwise comparisons, we can utilise these different datasets together as cross-dataset differences is removed.

Each dataset is capped at 10k rows to prevent larger datasets from dominating. Total training data after capping is roughly around 22k training pairs. The 3 main benchmarks used for evaluation are **HaHa pairwise** (N=2000, 20% split from hahackathon) and **NYCC pairwise** (N=1020, fold0 test+val, human agreement 64.6%) and Humicroedit pairwise (N=2628). (todo: wait i sort of forgot what this is about. did we use full humicro or something?) (todo: also is this a repeat with the experiements section?)

## 4.3 Considered but Excluded

There are several datasets considered but not included due to various reasons.
- **humor\_arena** (~2.5k pairwise, LLM-generated jokes): a LoRA trained on humor\_arena scored 52.8% cross-domain on HaHa pairwise, below the zero-shot baseline (-2.6pp), so it was excluded from joint training.
- **Jester** (dataset 3): only 140 unique jokes (though rated by 54k users). While pairwise pairs could be sampled abundantly, the content diversity is too low — with just 140 unique texts, the model would memorize item-level ordering rather than learn a generalizable humor signal.
- **Open Mic** (Mittal et al.): stand-up transcripts with audience laughter as ground truth; not publicly accessible.
- **Cards Against Humanity (CAH Lab)**: requires contacting the lab directly; did not respond.
- **Yelp Funny Reviews**: not explored due to time constraints.
- **Oogiri-GO** (Murakami et al.): not explored due to time constraints.
