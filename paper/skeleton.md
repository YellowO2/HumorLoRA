===========================================================
PAPER SKELETON — logical beats + planned citations per section
Workshop paper, 4–12 pages

PAPER FRAMING: Empirical study — "what works for giving LLMs humor taste?"
We tried many approaches (prompting, CoT, SFT, DPO, persona simulation,
probing, fine-tuning) and document what fails and why, what works and why.
A secondary contribution is assembling + unifying 5 crowd-labeled humor
datasets into a reusable pairwise training collection.
===========================================================

─── 1. ABSTRACT ────────────────────────────────────────────
Beat 1: Problem — no cheap automatic way to give or measure humor taste in LLMs
Beat 2: What we do — systematically evaluate approaches ranging from prompting
         to fine-tuning; assemble 5 crowd-labeled humor datasets unified into
         pairwise format as a reusable training collection
Beat 3: Key results — most prompting approaches plateau at 54–57% (near random);
         CoT and persona simulation actively hurt or add nothing; supervised
         fine-tuning on crowd-labeled pairwise data reaches 65–68%, near human ceiling
Beat 4: Takeaway — crowd supervision is the key ingredient; our dataset collection
         and findings provide a foundation for building humor-aware LLMs

─── 2. INTRODUCTION ────────────────────────────────────────
[1] Hook: want to make a humorous LLM — obvious directions (prompt, persona, theory)
[2] Reframe: bottleneck is selection, not generation — thought experiment (best-of-N)
    → empirically validated by SemEval 2026 Task 1 [CITE semeval2026 MWAHAHA]
[3] LLM-as-judge is the obvious selector — but requires humor sense to judge humor
    → subjective, hard to measure objectively, costly to use human crowd
[4] Define humor taste — weight vector over humor dimensions (incongruity, mechanism,
    target, language) [CITE Attardo & Raskin; Warren et al.]
[5] Good taste = crowd consensus — measurable via crowd-labeled pairwise datasets
[6] What we do — systematically test approaches; build a unified dataset collection;
    find that crowd-supervised fine-tuning is the only reliable path
[7] Preview of findings — key numbers (see Experiments)

─── 3. RELATED WORK ────────────────────────────────────────
Section story: prior work either evaluates per-dataset in isolation, studies humor
as classification (not preference), or proposes prompting tricks that we show fail.

Beat 1: Humor datasets and shared tasks
  - SemEval 2017 Task 6 — #HashtagWars [CITE Potash et al. 2017]
    Pairwise humor ranking; best acc = 67.5% with supervised ensemble (HumorHawk)
  - SemEval 2020 Task 7 — Humicroedit [CITE Hossain et al. 2020]
    Humor rating + pairwise; best pairwise 67.43% (Hitachi, fine-tuned ensemble)
  - SemEval 2021 Task 7 — HaHackathon [CITE Meaney et al. 2021]
    Humor rating + detection; 20 crowd annotators per item; no pairwise subtask
  - Hessel et al. 2023 — NYCC [CITE]
    Native pairwise captions; human ceiling 64.6%
  - Key observation: all prior work trains/evals per-dataset; we unify 5 datasets
    into one collection and study joint training

Beat 2: LLM prompting for humor judgment
  - LLM-as-judge [CITE Zheng et al. 2023 MT-Bench] — works for quality;
    known failure on subjective/cultural tasks including humor [CITE Verga et al. 2024 survey]
  - Persona simulation — Goes et al. 2022 Crowd Score [CITE]: 4 HSQ personality types vote
  - 17-dim humor basis — SemEval 2026 lmfaoooo [CITE]: score jokes on humor dimensions,
    aggregate; pointwise variant reaches only ~56%
  - We test all three empirically and find none outperforms zero-shot

Beat 3: CoT / reasoning for subjective tasks
  - Wang et al. 2025 [CITE] — CoT collapses judgment distribution in 30/40 scoring cases
  - We replicate: CoT hurts SHP -3.4pp (significant), neutral on NYCC
  - Interpretation: subjective preference has no ground-truth reasoning path;
    verbalizing a rationale introduces noise or bias

Beat 4: Supervised preference learning
  - Hu et al. 2021 LoRA [CITE] — parameter-efficient fine-tuning
  - Ouyang et al. 2022 InstructGPT [CITE] — pairwise preference training paradigm
  - Baranov et al. 2023 [CITE hri_tools] — cross-dataset transfer for humor classification;
    diversity aids generalization (binary labels only, not preference)
  - Santurkar et al. 2023 OpinionsQA [CITE] — LLMs misrepresent crowd views by default;
    fine-tuning on group data helps → motivates our supervised approach
  - We extend to pairwise preference training jointly across 5 datasets

─── 4. DATASETS ────────────────────────────────────────────
Section story: we assemble a collection of 5 diverse crowd-labeled humor datasets
spanning different domains, languages, and label types, and unify them into a
single pairwise training format. This collection is a reusable contribution.

Beat 1: Overview table
  | Dataset        | Domain                    | Label type         | Train rows    | Source paper         |
  | hahackathon    | short text jokes           | crowd rating 1–5   | ~3.9k         | Meaney et al. 2021   |
  | humicroedit    | news headline edits        | crowd rating 0–3   | ~10k (capped) | Hossain et al. 2020  |
  | reddit_jokes   | Reddit posts               | upvote ratio       | ~10k (capped) | Weller & Seppi 2019  |
  | haha_spanish   | Spanish text jokes         | crowd rating 0–4   | ~10k (capped) | Chiruzzo et al. 2019 |
  | nycc           | New Yorker cartoon capts   | native pairwise    | 2k pairs      | Hessel et al. 2023   |

Beat 2: Label harmonization — converting all to pairwise format
  - Rated datasets: sample (high-score, low-score) pairs from same source
  - Threshold criteria per dataset [mention score cutoffs]
  - Reddit: upvote_ratio as proxy; score ≥ 10 filter, dedup applied
  - NYCC: already pairwise; fold0 only, train/test/val non-overlapping
  - NYCC prompt includes image description (multimodal via text proxy)
  - Why pairwise: eliminates cross-dataset label scale differences;
    no normalization needed; same objective for all datasets

Beat 3: Cap and balance
  - Cap at 10k rows per dataset — prevents haha_spanish + humicroedit dominating
  - Total after capping: ~44k rows → ~22k training pairs
  - hahackathon is smallest (~3.9k) and used in full

Beat 4: Test sets (held out, never used in training)
  - HaHa pairwise: [N] pairs from hahackathon test split (20% hold-out)
  - NYCC pairwise: fold0_test + fold0_val ~1020 pairs; human ceiling 64.6% [CITE Hessel]
  - Note: Humicroedit and reddit_jokes have no held-out pairwise eval;
    we use HaHa and NYCC as the two primary benchmarks

Beat 5: Considered but excluded
  - humor_arena (~2.5k pairwise, LLM-generated jokes) — prepared and trained but excluded;
    binary 0/1 labels produced no useful signal (cross-domain acc near random 52.8%)
  - Jester (dataset 3) — 140 usable jokes rated by 54k users; too small for training,
    used only as an additional zero-shot eval point
  - Oogiri-GO [CITE Murakami et al.] — Japanese humor, 100 judges per prompt, no
    popularity bias; excluded due to language domain mismatch with our English datasets
  - Open Mic [CITE Mittal et al.] — stand-up transcripts with real audience laughter
    as behavioral ground truth (not annotation); not publicly accessible
  - Cards Against Humanity (CAH Lab) — naturalistic player choices, behavioral ground
    truth; requires contacting lab directly, not obtainable at time of writing
  - Yelp Funny Reviews — crowd "funny" votes on reviews; no public repository found

─── 5. EXPERIMENTS ─────────────────────────────────────────
Section story: systematic evaluation of approaches in order of complexity,
showing the wall that prompting hits and where supervision breaks through.

Sub-story A — WHAT FAILS (and why)

Beat A1: Zero-shot baseline — the wall
  - All base models: 54–57% on NYCC, HaHa pairwise, SHP, Jester
  - Consistent across architectures: Qwen3.5-4B, Hermes-3-8B, Gemma-4-4B, Llama-3.1-8B
  - Prompt framing irrelevant: gut / plain / crowd-framing all identical within noise
  - Interpretation: pretrained LLMs have no crowd-calibrated humor signal

Beat A2: CoT / thinking — hurts
  - SHP: -3.4pp (n=2000, significant)
  - NYCC: -1.3pp (within ±2.2pp CI, neutral)
  - Interpretation: subjective preference has no correct reasoning path;
    CoT either rationalizes noise or collapses distribution [CITE Wang et al. 2025]
  - Note: thinking = prompt-level CoT only (enable_thinking=False)

Beat A3: Style fine-tuning — hurts
  - Discord SFT: NYCC -5pp (Hermes), HaHa r=0.228→0.015, SHP -3.7pp
  - Cause: general reasoning degradation, not humor-specific
  - De-GPT DPO: flat across all 4 datasets (all within ±2.2pp noise)
  - Interpretation: style alignment data carries no crowd humor signal

Beat A4: Persona simulation — random
  - Crowd Score (Goes et al. 2022): ~53% on HaHa pairwise (n=1400, converged)
  - Essentially random — worse than zero-shot 55.4%
  - 4 personas on same LLM produce nearly identical outputs regardless of instruction
  - Reference: Santurkar 2023 — persona prompts have low steerability [CITE]

Beat A5: Humor basis prompting — no gain
  - 17-dim humor basis (lmfaoooo / SemEval 2026 approach): 56.1–56.3% on Humicroedit
  - No improvement over zero-shot 56.3%; decomposing into features adds no signal
  - Pointwise scoring collapses (model assigns same basis scores to most jokes)

Sub-story B — WHAT WORKS (and why)

Beat B1: Activation probing — linear funniness direction exists
  - Binary probe (funny/unfunny labels) on Humicroedit: 61.9% pairwise (layer 16)
  - Funniness is linearly represented in LLM activations [connects to RepE, CITE Zou et al. 2023]
  - But cross-domain transfer is weak: probe trained on Humicroedit → HaHa pairwise 55.5%
  - Margin-filtered probe (pairs with |score_A - score_B| ≥ 1.0): 67.0% on Humicroedit

Beat B2: Single-dataset LoRA — supervised signal is learnable
  - HaHa LoRA regression: 68.3% on HaHa pairwise — beats SemEval 2020 winner 67.43%
  - NYCC LoRA (2k pairs, 3 epochs): 65.7% on NYCC — near/above human ceiling 64.6%
  - In-batch pairwise BCE ≈ LoRA regression (67.7% vs 68.3%, within noise)
  - Each dataset teaches a domain-specific signal; cross-domain transfer is partial

Beat B3: Joint LoRA across 5 datasets — generalizes
  - HaHa pairwise: 65.2% | NYCC pairwise: 63.2%
  - Slight drop vs single-dataset (expected: generalization vs domain fit trade-off)
  - Key: within-source masking ensures only same-domain pairs compared in loss
  - Shows crowd labels from diverse sources share a common learnable humor signal

─── 6. ANALYSIS / DISCUSSION ───────────────────────────────
[Merge with Experiments if tight on space]

Beat 1: Why prompting plateaus
  - Pretrained LLMs encode a skewed humor sense (culturally dominant, not crowd-calibrated)
  - No prompt restructuring fixes this — the weights carry the prior
  - CoT makes it worse by introducing rationalization on a task with no ground-truth logic
  - Contrast with SHP (63% zero-shot): LLMs have implicit helpfulness preference from RLHF
    training on human feedback, but no equivalent signal for humor — motivates supervision

Beat 2: Why Crowd Score fails
  - HSQ types describe how people use humor in life, not dimensions of jokes (category error)
  - Calling same LLM 4× with personality instructions ≠ sampling from crowd distribution
  - Santurkar 2023: persona steerability is low; model defaults to its prior [CITE]

Beat 3: What crowd labels provide
  - Direct supervision on what crowds find funny, bypassing LLM's prior
  - Pairwise format removes label scale issues across datasets
  - Even noisy labels (reddit upvotes) contribute when combined with cleaner datasets

Beat 4: Limitations
  - All humor is crowd-aggregated; individual taste variation ignored
  - Spanish dataset included, no Spanish eval benchmark
  - NYCC: model sees image description, not actual image
  - Base model is 4B — larger models may learn differently

─── 7. CONCLUSION ──────────────────────────────────────────
Beat 1: Summary — tried 11 approaches; prompting plateau is real and robust;
         crowd-supervised fine-tuning reliably breaks through it
Beat 2: Dataset contribution — 5 unified crowd-labeled humor datasets in pairwise
         format; reusable for future humor preference research
Beat 3: Practical implication — joint LoRA as reward model in best-of-N generation loop
         (connects back to intro thought experiment)
Beat 4: Future work
  - RLHF loop: use this reward model to fine-tune a generator end-to-end
  - Larger base model
  - Vision input for NYCC (actual image, not description)
  - Cross-lingual eval using haha_spanish as eval set

===========================================================
PENDING / OPEN QUESTIONS
- SemEval 2026 task number: currently "MWAHAHA" per PROGRESS.md [verify exact citation]
- Crowd Score n=1400 partial result — confirm 53% is final/converged for paper
- HaHa pairwise test set size N [check datasets/hahackathon/pairwise.csv wc -l]
- Exact LoRA config (rank, alpha, target modules) [check train scripts]
- Baranov et al. 2023 exact citation [hri_tools paper — need to find arxiv link]
- Method section: confirm exact token logit extraction (which two tokens?)
===========================================================
