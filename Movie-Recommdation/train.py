# train.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load your movie dataset
# Replace 'movies.csv' with your actual movie dataset
movies_df = pd.read_csv('tmdb_5000_movies.csv')  # Must contain a 'title' and 'overview' column

# 2. Fill NaNs in overview
movies_df['overview'] = movies_df['overview'].fillna('')

# 3. Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['overview'])

# 4. Compute cosine similarity matrix
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 5. Save similarity matrix
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

# 6. Save movie titles with index
# This makes it easier to fetch movie names later
movies = movies_df[['title']]
with open('movies.pkl', 'wb') as f:
    pickle.dump(movies, f)

print("✅ Training complete. 'similarity.pkl' and 'movies.pkl' saved.")
