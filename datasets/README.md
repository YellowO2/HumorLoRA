# Datasets

Contains the datasets used in this project for evaluating LLM humor judgment.

## File structure
.
├── README.md
├── jester
│   ├── cleaned_ranked_dataset3.csv
│   ├── cleaned_ranked_dataset4.csv
│   ├── dataset3JokeSet.csv
│   ├── dataset3_ratings.csv
│   ├── dataset4JokeSet.csv
│   └── dataset4_ratings.csv
└── newyorker
    ├── ranking_fold0_test.csv
    ├── ranking_fold0_train.csv
    ├── ranking_fold0_validation.csv
    ├── ranking_fold1_test.csv
    ├── ranking_fold1_train.csv
    ├── ranking_fold1_validation.csv
    ├── ranking_fold2_test.csv
    ├── ranking_fold2_train.csv
    ├── ranking_fold2_validation.csv
    ├── ranking_fold3_test.csv
    ├── ranking_fold3_train.csv
    ├── ranking_fold3_validation.csv
    ├── ranking_fold4_test.csv
    ├── ranking_fold4_train.csv
    ├── ranking_fold4_validation.csv
    └── raw
- `jester/`: Contains the processed Jester joke dataset files. The base data features jokes rated by users. We transform these into pairwise comparisons where the model must choose the funnier joke (expected output: "A" or "B").
- `newyorker/`: Contains the New Yorker Caption Contest dataset splits. The task is to evaluate which caption is funnier for a given cartoon description. The datasets are restructured for a pairwise ranking task (expected output: "A" or "B").

## Preparation
We provide scripts to rebuild or prepare the datasets in `scripts/prepare_datasets/`:
- `prepare_jester_dataset.py`: Cleans and ranks the Jester jokes.
- `prepare_newyorker_dataset.py`: Downloads and processes the New Yorker dataset into folds with the expected target set to "A" or "B".

Note: When evaluating, the target `expected` field will strictly map to `A` or `B` to match the LLM prompt's generation structure.

