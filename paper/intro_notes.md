# Introduction Notes

## User's Raw Reflection (basis for intro)

Original motivation: build an LLM agent that posts content online and grows a social media following.
Reframed for paper: building an AI that generates engaging content raises the question of how to
programmatically evaluate whether AI-generated content is actually funny.

Manual evaluation is tedious, subjective, and slow. This led to wanting an LLM-as-judge.
But LLMs have notoriously poor humor sense — so how do you know the judge is accurate?
This is the core problem: you need a judge to avoid manual evaluation, but you need to
validate the judge somehow. Not circular — just: the bottleneck isn't generation, it's evaluation/selection.

### The Thought Experiment (keep this — connects to best-of-N / RLHF)
Give a generative LLM a prompt, sample 100 responses at high temperature.
If you can pick the funniest one, the result is likely quite funny.
Add a refinement loop (refine on funniest, create variations, select again).
This is how humans make jokes — generate candidates internally, filter via sense of humor.
A "sense of humor" for AI = reward model for humor = the key missing piece.

### The "Taste" Framing (original conceptual contribution)
Humor sense is one instance of "taste" — a subjective weighting function over objective features.
Example: taste in food = objective measurements (salt, sugar, etc.) + subjective weights per person.
Humor = f(X1, X2, ..., Xn) where Xi are humor dimensions (intelligence, surprise, wordplay, etc.)
(Ruch 1996 treats sense of humor as multidimensional trait like intelligence or temperament)
(SemEval 2026 MWAHAHA operationalizes this as 17 humor dimensions)

What is "good" taste for humor? → taste that correlates with crowd consensus / majority preference.
This gives us an objective definition: a good humor judge correctly predicts what the crowd finds funnier.
This definition is measurable via crowd-labeled humor datasets.

### Proposed Intro Flow
1. Motivation: building a funny AI → need to evaluate funniness automatically
2. LLM-as-judge seems like the answer → but LLMs have poor humor sense (cite Verga et al. 2024)
3. Bridge: the bottleneck isn't generation, it's evaluation/selection (thought experiment)
4. "Sense of humor" as a reward model / taste for humor
5. What is "good" taste? → crowd consensus → measurable
6. Therefore: we study whether LLMs can be given a sense of humor, using crowd-labeled data as ground truth
7. Empirical anchor: all prompting approaches plateau at 54–57%; SemEval winners at 67%+ via supervised training

### Key References for Intro
- Verga et al. (2024) — LLM-as-judge survey, known biases, humor as subjective/cultural failure case
- Wang et al. (2025) — CoT collapses judgment, corroborates our SHP finding
- Hessel et al. (2022) — NYCC, human ceiling 64.6%
- Kirk et al. (2024) — human preferences are diverse and cross-cultural, crowd aggregation needed
- Ruch (1996) — sense of humor as multidimensional trait
- SemEval 2026 MWAHAHA — 17-dimension humor basis

## Pending Decisions
- Whether to mention social media origin explicitly (currently: yes, briefly reframed)
- Whether to include the "taste" section in intro or move to related work / discussion
- Abstract to be written last
