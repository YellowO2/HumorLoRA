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
When I see an LLM persona online, first impression is, uuUUUuuuuugghh get that thing away! But, suppose you want to make one that is less hated, having a sense of humor is certainly an important factor. Perhaps I can do some prompt engineering to make a humorous AI, like giving it a persona of a comedian, grounding it in theory of humor, giving it some examples, then finally, judge if its output becomes funnier.

Wait, but how do I judge if it is becoming funnier? After all, humor is highly subjective (Castro et al., 2018; Chiruzzo et al., 2019). I could find I'm getting the LLM funnier, but Mr Joe says otherwise. The standard for subjective things is to recruit a crowd and rate the outputs, BUT Ain't nobody got time for that man! Hmmm, what comes to mind as replacements for people then? ....... It's goddamn AI of course!!! We can use LLM as a judge! 

This sounds good, however, notice this just pushes the problem back again. How do we know if the LLM Judge itself even has a good sense of humor? Can we measure it? To do that, we first need to define sense of humor itself, then what makes one "good."

We can think of sense of humor as a taste for humor. Each person's taste is a unique weighted combination of features like incongruity, logical mechanism, target, and language (Attardo & Raskin, 1991; Warren & McGraw, 2016):  w1.f1 + w2.f2 + ... that outputs how funny something is to them.

Individual tastes for humor vary, but they cluster around a center, analogous to how people have different taste for food but broadly agree on what's good. We can therefore define a "good" sense of humor as one that aligns with the crowd: given a pairwise choice, a good judge picks what more people find funnier. This definition lets us measure sense of humor objectively, by using crowd-labeled datasets.

In fact, it can be argued that this good sense of humor is all you need for a funny LLM. Consider this thought experiment: Prompt an LLM to generate 1000 joke attempts at high temperature. If you can pick the funniest response, the result is likely quite funny. This strategy is validated by the SemEval 2026 Task 1 (MWAHAHA) winner, where they framed humor generation as a selection problem, not a generation one. 

We therefore study whether an LLM judge can acquire this good sense of humor, and what approaches actually work. We evaluate approaches spanning prompting, CoT, persona simulation, crowd simulation, activation probing, and fine-tuning. Such a judge can then further serve as a reward model to train a humorous LLM directly, bypassing the selection step.

Key findings: (1) prompting-based approaches all plateau at 54–57% regardless of framing, CoT, few-shot (5 examples), or model size, and (2) crowd-supervised LoRA fine-tuning reliably improves over zero-shot. A single-dataset LoRA matches average human accuracy on NYCC (65.7%), and a joint LoRA across all five datasets achieves 65.2% on HaHa and 63.2% on NYCC.