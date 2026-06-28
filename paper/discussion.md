
[todo what is the point of our discussion section and intuition? need to rethink]

# Discussion

Humor is highly subjective. Studies measuring inter-annotator agreement on funniness ratings find Krippendorff's alpha of 0.1625 and 0.224 across two iterations of the same task (Castro et al., 2018; Chiruzzo et al., 2019), indicating low inter-rater agreement on individual funniness judgments. Yet individual variation does not mean anything goes. Much like how people disagree on their favorite food but broadly converge on what is generally good, humor preferences follow a population-level distribution with a learnable center. On NYCC, individual humans agree with the crowd majority 64.6% of the time (Hessel et al., 2023), confirming that a stable crowd signal exists even when individual taste varies widely. Fine-tuning on crowd labels reaches 65.7% on the same benchmark, matching individual human performance against crowd consensus. [todo, is this a repeat? i think we said some of these in previous section]

## 6.1 Why Prompting-Based Approaches Plateau

The zero-shot plateau at 54–57% is not a coincidence or a model-size issue. It reflects a structural gap: LLMs are preference-trained on helpfulness (via RLHF on human feedback), which is why they reach ~63% zero-shot on SHP. No equivalent crowd-calibrated signal exists for humor in standard pretraining or alignment pipelines. The model has opinions about what is funny, but those opinions are not anchored to what crowds actually find funny.

Prompt restructuring cannot fix this. Changing framing from gut-feeling to crowd-prediction to humor-expert persona all produce the same result within noise, because the weights carry the prior and the prior is uncalibrated. Chain-of-thought makes things worse: humor preference has no ground-truth reasoning path, so asking the model to verbalize a rationale either rationalizes noise or forces a confident answer on a question that is inherently probabilistic. Wang et al. (2025) observe the same collapse in 30/40 scoring scenarios; our results replicate it in humor.

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
