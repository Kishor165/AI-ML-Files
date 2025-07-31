from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

app = Flask(__name__)

# Global variables
movies_df = None
cosine_sim = None

def safe_literal_eval(val):
    if pd.isna(val) or val == '':
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def extract_names(data, key='name', limit=3):
    if isinstance(data, list):
        return [item.get(key, '') for item in data[:limit] if isinstance(item, dict)]
    return []

def preprocess_data():
    global movies_df, cosine_sim

    try:
        movies_df = pd.read_csv('tmdb_5000_movies.csv')
        movies_df = movies_df.fillna('')

        for col in ['genres', 'keywords', 'production_companies']:
            if col in movies_df.columns:
                movies_df[col] = movies_df[col].apply(safe_literal_eval)

        movies_df['genre_names'] = movies_df['genres'].apply(lambda x: extract_names(x, 'name', 5))
        movies_df['keyword_names'] = movies_df['keywords'].apply(lambda x: extract_names(x, 'name', 10))
        movies_df['company_names'] = movies_df['production_companies'].apply(lambda x: extract_names(x, 'name', 3))

        movies_df['combined_features'] = (
            movies_df['genre_names'].apply(lambda x: ' '.join(x) if x else '') + ' ' +
            movies_df['keyword_names'].apply(lambda x: ' '.join(x) if x else '') + ' ' +
            movies_df['overview'].fillna('') + ' ' +
            movies_df['company_names'].apply(lambda x: ' '.join(x) if x else '')
        )

        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        print(f"Loaded {len(movies_df)} movies.")
        return True

    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def get_recommendations(title, num=10):
    try:
        idxs = movies_df[movies_df['title'].str.lower() == title.lower()].index
        if len(idxs) == 0:
            # partial match fallback
            partials = movies_df[movies_df['title'].str.lower().str.contains(title.lower(), na=False)]
            if partials.empty:
                return None, "Movie not found in database"
            idx = partials.index[0]
        else:
            idx = idxs[0]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in sim_scores[1:num+1]]

        recommendations = []
        for i in movie_indices:
            movie = movies_df.iloc[i]
            recommendations.append({
                'title': movie['title'],
                'overview': (movie['overview'][:200] + '...') if len(movie['overview']) > 200 else movie['overview'],
                'genres': ', '.join(extract_names(movie['genres'], 'name', 3)),
                'release_date': movie.get('release_date', 'N/A'),
                'vote_average': movie.get('vote_average', 'N/A'),
                'popularity': round(movie.get('popularity', 0), 1)
            })

        return recommendations, None

    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie_title', '').strip()
    if not movie_title:
        return render_template('recommend.html', error="Please enter a movie title")

    recommendations, error = get_recommendations(movie_title)
    if error:
        return render_template('recommend.html', error=error, movie_title=movie_title)

    return render_template('recommend.html', recommendations=recommendations, movie_title=movie_title)

if __name__ == '__main__':
    print("Loading data...")
    if preprocess_data():
        print("Starting app...")
        app.run(debug=True)
    else:
        print("Failed to load data.")
