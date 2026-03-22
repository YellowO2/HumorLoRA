# Datasets

About the datasets used in this project.

## Jester Dataset

### Dataset 3
- Around 2.3 million ratings.
- Rating range: -10.00 to +10.00.
- Contains 150 jokes.
- Includes historical data collected across multiple periods.

Format notes:
- Joke text is provided as an Excel sheet with 150 rows.
- Row number maps to joke ID.
- Ratings data is provided separately in zipped files.
- Ratings table has users as rows and jokes as columns, with a leading column for number of jokes rated by each user.

### Dataset 4
- Over 100,000 newer ratings.
- Includes 8 additional jokes (IDs 151-158).
- Joke text is provided in zipped files.

Format notes:
- Excel sheet has 158 rows.
- First 150 joke IDs are consistent with prior datasets.

### cleaned_ranked_datasets
- the cleaned and ranked dataset, where the 100+ jokes are ranked by their average rating from most funny to least funny.