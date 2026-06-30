
# Discussion

Here we discuss why prompting fails to access it and why fine-tuning on crowd labels succeeds.

## 6.1 Why Prompting-Based Approaches Plateau

Prompting is often seen as a powerful and flexible tool: you can change framing, add personas, provide examples, or ask the model to reason step by step. Yet all variants we tested plateau at the same level. An interpretation is that "taste", as we theorised, is a high-dimensional weighted sum consisting of many factors. It is hence very difficult to capture or shift the model to behave correctly across many of these factors at once, as we don't even know clearly how many or what these factors are. In fact, this is why we often describe it as "taste": we can't even verbalise it ourselves. Hence, trying to use prompting to nudge this preference into a correct one is very unlikely to work. Crowd Score (Goes et al., 2022) fails for the same reason: prompting the same model four times with different humor personas cannot simulate a crowd's taste when the underlying weights are unchanged. By contrast, the same model reaches 63.2% on SHP zero-shot, because RLHF did directly train it on human helpfulness preferences, showing that preference training, not prompting, is what provides the calibration.

Another interesting point is specific to CoT, which improves most tasks, but not here. Building on the above, since "taste" has so many factors, there is no real reasoning path and it is more like a gut reaction. Asking the model to reason through it creates the sharpening effect proposed by Wang et al. (2025), where the model commits to one reasoning trace, and since there is no real basis for that path, committing to it just increases the chance of being wrong.

## 6.2 What Crowd Labels Provide

Crowd-labeled pairwise data works because it directly supervises the thing that matters: what a population finds funnier, not what the model thinks or what any single annotator thinks. Three properties of our setup contribute to this.

First, the pairwise format sidesteps the inter-dataset label scale problem. A rating of 2/5 on hahackathon and 1/3 on humicroedit are not comparable, but the ordering within each source is. Converting to (funnier, less funny) pairs allows a single training objective across all five datasets without normalization.

Second, even noisy crowd labels contribute signal. Reddit upvote ratio is an imperfect proxy for funniness, yet the reddit\_jokes LoRA achieves the strongest cross-domain transfer to HaHa (+3.5pp), suggesting the noise is not so overwhelming that the crowd tendency gets lost.

Third, crowd preference signals from diverse humor domains are compatible. Training jointly on five datasets (Spanish jokes, news headline edits, Reddit posts, cartoon captions, and English text jokes) loses only a few points versus single-dataset LoRAs while generalizing across both benchmarks simultaneously. This is consistent with a view that there is a shared learnable direction in the model's representation space corresponding to broad humor preference, one that crowd labels from different domains all point toward, even if imperfectly.

## 6.3 Limitations

- no verification that this sense of humor trained is truely out of domain humor and true sense of humor 

Our evaluation targets crowd-aggregated preference, which smooths over individual taste variation. A model that fits the crowd average well may still fail to match any specific person's sense of humor.

For NYCC, we substitute a text description of the cartoon image rather than the actual image. The model never sees the visual content that human annotators saw, which likely limits NYCC performance below what a multimodal model could achieve.

Our largest model tested is with 9B parameters. Larger models may respond differently to the same supervision.
