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
Consider this thought experiment: give a generative LLM a prompt, sample 100 responses at high temperature. If you can pick the funniest one, the result is likely quite funny. Add a refinement loop — refine on the funniest, create variations, select again. This is how humans make jokes internally. The bottleneck, then, is not generation — it is selection. A "sense of humor" for AI is the missing piece: a reward model that can reliably pick the funnier output.

[3 — Why LLM-as-judge fails]
The obvious candidate for this selector is an LLM-as-a-judge — this has become popular and works well for certain more objective tasks. But this just pushes the problem back: it requires the judge to already have a good sense of humor. And a big problem is: how are we supposed to tell that a technique is actually making the LLM more or less funny? After all, humor is highly subjective (cite). We could easily feel we are getting the LLM funnier, or less funny, but it's hard to claim objectively. Typically, what is done is utilizing a human crowd to evaluate, which is costly and slow. Anyone who has asked an LLM to say something funny will feel that they have a bad sense of humor — but how do we prove this objectively? What does it even mean to have a good sense of humor, and how do we measure it?

[4 — Define sense of humor / taste]
We believe that this "taste" for humor is an important concept. Taste is really just a weight vector over a set of more observable features that differs from person to person. In the case of humor, we can break it down into many different features — dimensions like incongruity, logical mechanism, target, language (cite Attardo & Raskin; Warren et al.) — and then taste is simply a weighted sum of all these factors that returns an output of how funny you feel. This is your sense of humor.

[5 — Define good taste = crowd consensus]
Hence there exists some weight vector that is the mean — the crowd's "taste" for humor. We define a good sense of humor as one that agrees with the crowd: if we say an LLM judge has a good sense of humor, it should agree with the crowd on what is funny and how funny. This definition is measurable via crowd-labeled humor datasets.

[6 — Therefore: what this paper does]
[placeholder — needs: "We therefore study whether LLMs can be fine-tuned to acquire this crowd-aligned humor sense, training on N crowd-labeled datasets and evaluating on pairwise benchmarks."]

[7 — Preview of findings]
[placeholder — needs result numbers once joint training is done.]