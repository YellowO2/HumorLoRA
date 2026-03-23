import pandas as pd
import numpy as np
from pathlib import Path

MISSING_RATING = 99

def prepare_and_save(dataset_num: int):
    print(f"Processing Dataset {dataset_num}...") 
    base_path = Path(__file__).parent.parent.parent / "datasets" / "jester" #use relative path
    
    # 1. Load jokes
    jokes_file = base_path / f"dataset{dataset_num}JokeSet.csv"
    jokes_df = pd.read_csv(jokes_file, header=None, quotechar='"')
    jokes = jokes_df[0].tolist()
    num_jokes = len(jokes)
    
    # 2. Load ratings (skip first column which is "count of jokes rated by user")
    ratings_file = base_path / f"dataset{dataset_num}_ratings.csv"
    ratings_df = pd.read_csv(ratings_file, header=None)
    ratings = ratings_df.iloc[:, 1:num_jokes + 1].values
    num_users = ratings.shape[0]
    
    # 3. Clean out-of-bounds ratings
    valid_mask = ratings != MISSING_RATING
    out_of_bounds = valid_mask & ((ratings < -10.0) | (ratings > 10.0))
    ratings[out_of_bounds] = MISSING_RATING
    
    print(f"  - Loaded {num_jokes} jokes and {num_users} users.")
    print(f"  - Cleaned {out_of_bounds.sum()} out-of-bounds ratings.")
    
    # 4. Calculate stats for each joke
    stats = []
    min_rating_count = 100
    for joke_id in range(num_jokes):
        joke_ratings = ratings[:, joke_id]
        valid_ratings = joke_ratings[(joke_ratings != MISSING_RATING) & (~np.isnan(joke_ratings))]
        
        if len(valid_ratings) >= min_rating_count:
            stats.append({
                'joke_id': joke_id + 1,
                'avg_rating': float(np.mean(valid_ratings)),
                'num_ratings': len(valid_ratings),
                'std_dev': float(np.std(valid_ratings)),
                'joke_text': jokes[joke_id]
            })
            
    # 5. Create DataFrame, sort, and rank
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('avg_rating', ascending=False)
    stats_df['rank'] = range(1, len(stats_df) + 1)
    
    # Reorder columns
    final_df = stats_df[['rank', 'joke_id', 'avg_rating', 'num_ratings', 'std_dev', 'joke_text']]
    
    # 6. Save to CSV
    output_path = base_path / f"cleaned_ranked_dataset{dataset_num}.csv"
    final_df.to_csv(output_path, index=False)
    print(f"  - Saved clean, sorted data to {output_path.name}\n")

if __name__ == "__main__":
    # Process both datasets
    prepare_and_save(3)
    prepare_and_save(4)
