# Patch
# Group 3
# Spotify Data Mining Project

import pandas as pd
import numpy as np
import ast

def load_data(file_path, rows=100000):
    """
    PURPOSE: Loads subset of data for speed during analysis
    """
    print(f"--- Loading {rows} rows from {file_path} --")
    return pd.read_csv(file_path, nrows=rows)

def data_review(df):
    """
    Purpose: Check to see if the data is ready for modeling
    """
    # 1. Data Shape and Info
    print("\n--- 1. Data Structure ---")
    print(df.info())

    # 2. Missing Data Check
    print ("\n --- 2. Missing Values ---")
    missing = df.isnull().sum()
    percent_missing = (missing / len(df)) * 100
    report = pd.DataFrame({'Missing': missing, '%': percent_missing.round(2)})
    print (report[report['Missing']>0])     # only show col. that have missing data

    # 3. Descriptive Stats: Mean, Median, Min, Max, SD
    print("\n--- 3. Numerical Summary & Outlier Detection ---")
    num_cols = [
            'danceability','energy','loudness','speechiness','acousticness',
            'instrumentalness','liveness', 'valence', 'tempo','popularity',
            'duration_ms','year','total_artist-followers'
        ]
    valid_num_cols = [c for c in num_cols if c in df.columns]

    # Transposing (.T) for readability
    stats = df[valid_num_cols].describe(percentiles=[.01,.25,.5,.75,.99]).T
    print(stats[['min','1%','50%','99%','max']])      # Highlighting tails

    # 4. Popularity Distribution 
    print("\n--- 5. Popularity Distribution (skewness) ---")
    print(f"Mean: {df['popularity'].mean():.2f}")
    print(f"Median: {df['popularity'].median()}")
    
    # Check for songs with 0 popularity (potential noise/missing data)
    zero_pop = (df['popularity'] == 0).sum()
    print (f" Number of song with 0 popularity: {zero_pop} ({(zero_pop/len(df)*100):.2f}%)")


    # 5. Music Theory 
    # normalize=True gives us percentage
    print("\n--- 4. Music Theory Distributions ---")
    if 'mode' in df.columns:
        print(f"Mode (0=Minor,1=Major):\n{df['mode'].value_counts(normalize=True)}")
    if 'key' in df.columns:
        print(f"Top 3 Musical Keys: \n{df['key'].value_counts(normalize=True).head(3)}")

    # 6. Feature Correlations
    # corr. = 1.0 is perfect match, -1.0 is perfect opposite
    print("\n--- 5. Top Features Correlated with Popularity ---")
    corr = df[valid_num_cols].corr()
    print(corr['popularity'].sort_values(ascending=False))

    # 7. Genre Complexity
    if 'genre' in df.columns:   # if high diversity might need to group them
        print("\n--- 4. Top 10 Genres ---")
        print(df['genre'].value_counts().head(10))
    if 'niche_genres' in df.columns:
        # check avg no. of niche tags per song
        def count_tags(x):
            try:
                # ast.literal_eval used b/c of spotify's single-quoted string lists
                tags = ast.literal_eval(x)
                return len(tags)
            except:
                return 0
        df['niche_tag_count'] = df['niche_genres'].apply(count_tags)
        print(f"\nAverage Niche Tags per Song: {df['niche_tag_count'].mean():.2f}")
        print(f"\nMax Niche Genres on one song: {df['niche_tag_count'].max()}")

    if __name__ == "__main__":
    dataset_path = 'songs.csv'
    try:
        df = load_data(dataset_path)
        data_review(df)
    except FileNotFoundError: 
        print(f"Error: {dataset_path} not found in current directory.")
   
