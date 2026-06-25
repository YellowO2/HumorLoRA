# Dataset Registry

| Dataset | Link | Size | Label | Notes |
|---------|------|------|-------|-------|
| humicroedit | https://github.com/n-hossain/semeval-2020-task-7-humicroedit | ~7MB | `meanGrade` 0–3 | Official train/test split |
| hahackathon | https://github.com/NLP-UMUTeam/SemEval2021-HaHackathon-UMUTeam/tree/main/datasets | ~1MB | `humor_rating` 0–5 | 80/20 split via `prepare/hahackathon.py` |
| jester | https://eigentaste.berkeley.edu/dataset/ | ~36MB | avg rating −10→+10 | Only 150 jokes; eval only |
| newyorker | https://huggingface.co/datasets/jmhessel/newyorker_caption_contest | ~17MB | pairwise caption winner | Images missing; using text description |
| humor_arena | https://github.com/SaveTheRbtz/humor | 2,541 entries (LEFT=594, RIGHT=619, NONE=1186, BOTH=142) | pairwise A/B | NONE/BOTH = ties; drop for accuracy eval, usable for regression training |
| humor_mechanics | https://github.com/altsoph/humor-mechanics | 120 jokes | `funniness` 1–5 | Too small to train; eval/analysis only |
| reddit_jokes | https://github.com/orionw/RedditHumorDetection/tree/master/full_datasets/reddit_jokes | 9.5MB / 83k rows | `score`, `upvote_ratio` (vote-based, noisy) | Needs normalization before use; score distribution very skewed |
| haha_spanish | https://www.fing.edu.uy/inco/grupos/pln/haha/2019/ | 24k train / 9k test | `funniness_average` 1–5 (NaN if not humorous) | Spanish tweets; 9,253 humorous with continuous scores + 14,747 non-humorous (label=0); Chiruzzo et al. (2019) |
