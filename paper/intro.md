1. Abstract — what you did, why, key result
2. Introduction — motivation, problem, contributions
3. Related Work — prior humor detection, RepE/probing, LoRA alignment
4. Datasets — what you collected and why
5. Method — in-batch pairwise LoRA, prompt design, joint training
6. Experiments — individual LoRA results, compatibility check, joint result
7. Analysis — what the cosine similarity told us, generalization
8. Conclusion

# Intro
---
[1 — Hook]
Suppose you want to make an LLM that generates engaging online content, humor is definitely an important factor. One might wonder: how do you make a humorous LLM? Perhaps via prompt engineering, giving it a persona, or grounding it in theory of humor? And hence we started out researching this.

[2 — Reframe: bottleneck is selection]
Consider this thought experiment: give a generative LLM a prompt, sample 100 responses at high temperature. If you can pick the funniest one, the result is likely quite funny. Add a refinement loop — refine on the funniest, create variations, select again. This is how humans make jokes internally. Indeed, this kind of best-of-N selection strategy is empirically validated: top systems in SemEval 2026 Task X [CITE semeval2026] relied on LLM-based selection over generated candidates to achieve state-of-the-art humor generation. The bottleneck, then, is not generation — it is selection. A "sense of humor" for AI is the missing piece: a reward model that can reliably pick the funnier output.

[3 — Why LLM-as-judge fails]
The obvious candidate for this selector is an LLM-as-a-judge — this has become popular and works well for certain more objective tasks. But this just pushes the problem back: it requires the judge to already have a good sense of humor. And a big problem is: how are we supposed to tell that a technique is actually making the LLM more or less funny? After all, humor is highly subjective (cite). We could easily feel we are getting the LLM funnier, or less funny, but it's hard to claim objectively. Typically, what is done is utilizing a human crowd to evaluate, which is costly and slow. Anyone who has asked an LLM to say something funny will feel that they have a bad sense of humor — but how do we prove this objectively? What does it even mean to have a good sense of humor, and how do we measure it?

[4 — Define sense of humor / taste]
We believe that this "taste" for humor is an important concept. Taste is really just a weight vector over a set of more observable features that differs from person to person. In the case of humor, we can break it down into many different features — dimensions like incongruity, logical mechanism, target, language (cite Attardo & Raskin; Warren et al.) — and then taste is simply a weighted sum of all these factors w1.f1 + w2.f2 ... that returns an output of how funny you feel. This is your sense of humor.

[5 — Define good taste = crowd consensus]
Hence there exists some weight vector that is the mean — the crowd's "taste" for humor. We define a good sense of humor as one that agrees with the crowd: if we say an LLM judge has a good sense of humor, it should agree with the crowd on what is funny and how funny. This definition then allows us to objectively measure how good our judge is performing, via utilisation of crowd-labeled humor datasets.

[6 — Therefore: what this paper does]
We therefore study whether LLMs can be fine-tuned to acquire this crowd-aligned sense of humor. Concretely, we train LoRA adapters [CITE Hu et al. 2021] on five crowd-labeled humor datasets spanning text jokes, social media content, and image captions — using a pairwise preference objective that directly encodes crowd judgments. We evaluate on held-out pairwise benchmarks and compare against zero-shot baselines and the Crowd Score method of Goes et al. (2022) [CITE], which simulates crowd preferences via four LLM humor personalities.

[7 — Preview of findings]
We find that: (1) a single-dataset LoRA can reach near-human pairwise accuracy on its own benchmark (65.7% on NYCC vs. human ceiling ~64.6%); (2) a joint LoRA trained across all five datasets generalizes across domains, achieving 65.2% on HaHa pairwise and 63.2% on NYCC; and (3) the Crowd Score personality-simulation baseline achieves [TBD]%, suggesting that role-playing humor archetypes is [a weak / a competitive] proxy for genuine crowd taste. Our results suggest that crowd-labeled pairwise data, combined with lightweight fine-tuning, is a practical path toward LLMs with measurable, generalizable humor sense.