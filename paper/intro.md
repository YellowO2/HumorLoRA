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
When I see an LLM persona online, first impression is, ewwwww get that shit away! But, suppose you want to make one that is less hated, having a sense of humor is certainly an important factor. Perhaps I can do some prompt engineering to make a humorous AI, like giving it a persona of a comedian, grounding it in theory of humor, giving it some examples, then finally, judge if its output becomes funnier.
Wait, but how do I objectively judge if it is becoming funnier? After all, humor is highly subjective (Castro et al., 2018; Chiruzzo et al., 2019). I could easily feel I'm getting the LLM funnier, or less funny, but Mr Joe says otherwise. The standard for subjective things is to recruit a crowd and rate the outputs, BUT Ain't nobody got time for that man!
What comes to mind as replacements for people? ---- it's fUCKING AI!!! I can use LLM as a judge! -- Or can I? Because this just pushes the problem back again: How do I know if the Judge even has a good sense of humor?

Hence, we need to, 1, be able to measure sense of humor, then 2, make an LLM with good sense of humor. [todo]

Which brings out the KEY question. How do you even measure sense of humor? What is a good sense of humor? I think LLMs have a bad sense of humor, but how do I prove this objectively?

First, we would need to define what we mean by a "good" "sense of humor".

We can think of sense of humor as a "taste" for humor — a weighted vector over features like incongruity, logical mechanism, target, and language (Attardo & Raskin, 1991; Warren & McGraw, 2016), where each person's taste is a unique combination w1.f1 + w2.f2 + ... that outputs how funny something feels to them.

If individual tastes can be represented as weight vectors, then there exists some mean vector representing the crowd's taste, analogous to how people have different food preferences but broadly agree on what's good. This is empirically observable in crowd-labeled humor datasets.

We can therefore define a "good" sense of humor as one that aligns with the crowd: given a pairwise choice, a good judge picks what more people find funnier. This definition lets us measure humor sense objectively using crowd-labeled datasets.

In fact, I would argue that having a GOOD sense of humor is all you need for a funny LLM. Consider this thought experiment: give a generative LLM a prompt, sample 1000 responses at high temperature. If you can pick the funniest response, the result is likely quite funny. Not to mention a refinement loop can be added. Refine on the funniest, create variations, select again, similar to how humans make jokes internally. This kind of best-of-N selection strategy is also validated by the SemEval 2026 Task 1 (MWAHAHA) winner, where they relied on LLM-based selection over generated candidates to achieve state-of-the-art humor generation. Furthermore, an LLM judge with a good sense of humor can also be used as a reward model to fine-tune a humorous LLM directly, bypassing the need for best-of-N selection at inference time.

We therefore study whether LLMs can acquire this good sense of humor, and what approaches actually work. Concretely, we evaluate approaches spanning prompting, CoT, persona simulation, crowd simulation, activation probing, and fine-tuning, and find that most fail to improve over zero-shot, while strongest results come from LoRA finetuning on crowd labeled humor datasets.

Key findings: (1) prompting-based approaches all plateau at 54–57% regardless of framing, CoT, few-shot (5 examples), or model size, and (2) crowd-supervised LoRA fine-tuning reliably improves over zero-shot. A single-dataset LoRA matches average human accuracy on NYCC(65.7%), and a joint LoRA across all five datasets achieves 65.2% on HaHa and 63.2% on NYCC.
