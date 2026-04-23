## Quick recap

Patch presented an update on their data analysis project, sharing screen-sharing results including artist popularity distributions, genre-based categorizations, and audio feature correlations. The team divided artists into four categories based on median followers and popularity levels, combined certain music genres, and identified significant correlations between audio features like danceability, valence, and acousticness. Yanjun provided feedback on the analysis, suggesting improvements such as adding back the rock genre through random sampling, implementing proper train-test-evaluation splits with cross-validation, conducting feature selection, and addressing class imbalance in the popularity prediction task. The team also discussed signing up for a presentation slot on the 24th, with Paromita confirming the scheduling details.
## Next steps

- Patch: Update datasets to include 5,000 randomly sampled rock songs (instead of dropping the entire rock genre), ensuring balanced class sizes for modeling.
- Patch: Split datasets into three parts (train, evaluation, and test) for modeling, rather than just 80-20 split, and perform feature selection and normalization before modeling.
- Patch: Check class distribution for song popularity and, if unbalanced, consider using SMOTE or similar methods to balance the dataset.
- Paromita: Sign up the group for the presentation slot on the 24th.
- Patch/Team: Prepare and present findings using clearer, understandable descriptions of audio features in the final report/presentation.

## Summary

### Artist Data Analysis Updates

Patch shared updates on their data analysis work, including combining files and examining the artist dataset. They explained how they divided artists into four categories based on median followers and popularity levels, creating groups such as high followers/high popularity for mainstream artists. The analysis was presented on a log scale due to the large data range.
### Music Genre Popularity Analysis

Patch and Yanjun discussed the popularity distribution of songs and the decision to combine certain genres like jazz and blues, hip-hop and R&B, and country and folk. They considered dropping rock due to its large size but Yanjun suggested using random sampling to balance it instead. Patch presented audio feature correlations and a correlation matrix, explaining that certain features like danceability and instrumentalness were related to popularity, though some terms like "valence" required further clarification.
### Audio Features Analysis Discussion

Yanjun and Patch discussed the analysis of audio features in songs, focusing on correlations such as danceability, musical positiveness, energy, and acousticness. They clarified the definitions of terms like acousticness and discussed how to better describe these features in a presentation. They also reviewed how they categorized audio features by popularity tiers, noting differences in the grouping methods used for songs and artists.
### Popularity Scale Dataset Selection

Yanjun and Patch discussed the selection of numbers for a popularity scale and the process of creating datasets for classification and prediction tasks. Patch explained the use of standard deviation in selecting the numbers and described the two main prediction tasks: classifying genre based on audio features and predicting popularity levels (low, medium, or high). Yanjun advised on handling features, including removing outliers, addressing missing values, and normalizing data types, and suggested updating the dataset to include a balanced sample of the rock genre.
### Data Splitting and Feature Selection

Yanjun advised Patch to modify their data splitting approach, suggesting a three-part split instead of the current 80-20 split, with separate train, evaluation, and test sets. Yanjun also recommended implementing feature selection to improve model performance, particularly since some audio features like artist information were already removed from the 24 original features.
### Audio Project Focus Discussion

Yanjun and Patch discussed focusing on the audio part of their project, including popularity analysis, feature processing, normalization, and modeling. Yanjun advised paying attention to class distribution balance and suggested using SMOTE if the distribution is unbalanced. Paromita joined the conversation to confirm the presentation date, which was set for the 24th at 1:25 PM, and she agreed to sign up for the presentation slot.
