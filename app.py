import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Modern Glassmorphism Design with Dark Theme ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root variables */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --dark-bg: #0a0b1e;
    --card-bg: rgba(255, 255, 255, 0.05);
    --border-color: rgba(255, 255, 255, 0.1);
    --text-primary: #ffffff;
    --text-secondary: #a0a0a0;
}

/* Global styles */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Main container */
[data-testid="stAppViewContainer"] {
    background: var(--dark-bg);
    background-image: 
        radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.1) 0%, transparent 50%);
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Title styling */
h1 {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 2rem !important;
    letter-spacing: -0.02em;
}

/* Section headers */
h2, h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    margin: 2rem 0 1rem 0 !important;
}

h2 {
    font-size: 2rem !important;
    position: relative;
}

h2:after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 0;
    width: 60px;
    height: 3px;
    background: var(--accent-gradient);
    border-radius: 2px;
}

/* Glass card effect for main content */
.main .block-container {
    padding: 2rem;
    background: var(--card-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-color);
    border-radius: 24px;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    margin: 1rem;
}

/* Radio button container */
.stRadio > div {
    background: var(--card-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.stRadio > div > label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    margin-bottom: 1rem !important;
}

/* Radio options */
.stRadio div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.stRadio div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(102, 126, 234, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
}

.stRadio div[role="radiogroup"] label[data-checked="true"] {
    background: var(--primary-gradient);
    border-color: transparent;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
}

/* Select box styling */
.stSelectbox > div > div {
    background: var(--card-bg) !important;
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease;
}

.stSelectbox > div > div:hover,
.stSelectbox > div > div:focus-within {
    border-color: rgba(102, 126, 234, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.stSelectbox label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
}

/* Button styling */
.stButton > button {
    background: var(--secondary-gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.8rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(245, 87, 108, 0.3) !important;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(245, 87, 108, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* Recommendation cards */
.recommendation-card {
    background: var(--card-bg);
    backdrop-filter: blur(15px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.recommendation-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    border-color: rgba(102, 126, 234, 0.3);
}

.recommendation-card:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--accent-gradient);
}

/* Text styling */
p, div, span {
    color: var(--text-primary) !important;
}

/* List styling for recommendations */
ul {
    list-style: none;
    padding: 0;
}

li {
    background: var(--card-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
    transition: all 0.3s ease;
    position: relative;
}

li:hover {
    transform: translateX(8px);
    border-color: rgba(102, 126, 234, 0.4);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

li:before {
    content: '📚';
    margin-right: 0.8rem;
    font-size: 1.2rem;
}

/* Warning/info styling */
.stWarning, .stInfo {
    background: var(--card-bg) !important;
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem;
        margin: 0.5rem;
    }
    
    h1 {
        font-size: 2.5rem !important;
    }
    
    .stButton > button {
        width: 100%;
        margin: 1rem 0;
    }
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--dark-bg);
}

::-webkit-scrollbar-thumb {
    background: var(--primary-gradient);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--secondary-gradient);
}

/* Loading animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 2s infinite;
}

/* Success message styling */
.success-message {
    background: linear-gradient(135deg, rgba(46, 213, 115, 0.1) 0%, rgba(0, 184, 148, 0.1) 100%);
    border: 1px solid rgba(46, 213, 115, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    color: #2ed573;
}
</style>
""", unsafe_allow_html=True)

# App title with emoji
st.markdown("# 📚 Book Recommender System")

# Load data
books = pd.read_csv("books.csv")  # Must have Book_ID, Title, Author, Genre
ratings = pd.read_csv("ratings.csv")  # Must have User_ID, Book_ID, Rating

# Content-based setup
books['features'] = books['Title'] + " " + books['Author'] + " " + books['Genre']
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['features'])
content_similarity = cosine_similarity(tfidf_matrix)

# Ratings pivot: users as rows, books as columns
ratings_matrix = ratings.merge(books[['Book_ID', 'Title']], on='Book_ID').pivot_table(
    index='User_ID', columns='Title', values='Rating').fillna(0)

# User-based similarity matrix
user_similarity = cosine_similarity(ratings_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

# Method selection with updated titles and descriptions
method_options = {
    '🎯 Content-Based': "Find books similar to your favorite based on genre, author, and content",
    '👥 Collaborative Filtering': "Discover books loved by users with similar taste to yours",
    '🔮 Hybrid Approach': "Get the best of both worlds with combined recommendations"
}

st.markdown("### Choose Your Recommendation Style")
method = st.radio(
    "Select how you'd like to discover your next great read:",
    options=list(method_options.keys()),
    help="Each method uses different algorithms to find books you'll love"
)

# Clean method name for processing
clean_method = method.split(' ', 1)[1]  # Remove emoji

# Display method description
st.markdown(f"*{method_options[method]}*")

st.markdown("---")

book_titles = books['Title'].tolist()
user_ids = sorted(ratings['User_ID'].unique().tolist())

selected_book = None
selected_user = None

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Select User")
    selected_user = st.selectbox(
        "Choose your User ID:",
        user_ids,
        help="This helps us understand your reading preferences"
    )

with col2:
    st.markdown("### 📖 Select Book")
    selected_book = st.selectbox(
        "Choose a book you enjoyed:",
        book_titles,
        help="We'll find similar books based on your selection"
    )

# Center the recommend button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    recommend_clicked = st.button("✨ Get My Recommendations", use_container_width=True)

if recommend_clicked:
    with st.spinner("🔍 Finding perfect books for you..."):
        st.markdown("## 🌟 Your Personalized Recommendations")

        if clean_method == 'Content-Based':
            idx = books[books['Title'] == selected_book].index[0]
            similar_indices = content_similarity[idx].argsort()[::-1][1:4]
            
            st.markdown("*Based on the book you selected, here are similar titles you might enjoy:*")
            
            for i, book_idx in enumerate(similar_indices, 1):
                book_title = books.iloc[book_idx]['Title']
                book_author = books.iloc[book_idx]['Author']
                book_genre = books.iloc[book_idx]['Genre']
                
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">#{i} {book_title}</h4>
                    <p style="margin: 0; color: #a0a0a0; font-size: 0.9rem;">by {book_author} • {book_genre}</p>
                </div>
                """, unsafe_allow_html=True)

        elif clean_method == 'Collaborative Filtering':
            if selected_user not in user_similarity_df.index:
                st.warning("⚠️ User ID not found in our ratings database. Please select a different user.")
            else:
                sim_users = user_similarity_df[selected_user].sort_values(ascending=False)[1:]
                user_ratings = ratings_matrix.loc[selected_user]
                unrated_books = user_ratings[user_ratings == 0].index

                scores = {}
                for book in unrated_books:
                    book_ratings = ratings_matrix[book]
                    sim_scores = sim_users[book_ratings > 0]
                    ratings_vals = book_ratings[book_ratings > 0]
                    if not sim_scores.empty:
                        score = np.dot(sim_scores, ratings_vals) / sim_scores.sum()
                        scores[book] = score

                if scores:
                    st.markdown("*Based on users with similar preferences, we recommend:*")
                    recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    for i, (book, score) in enumerate(recs, 1):
                        # Get book details
                        book_info = books[books['Title'] == book].iloc[0]
                        
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4 style="margin: 0 0 0.5rem 0; color: #f5576c;">#{i} {book}</h4>
                            <p style="margin: 0; color: #a0a0a0; font-size: 0.9rem;">
                                by {book_info['Author']} • {book_info['Genre']}
                            </p>
                            <p style="margin: 0.5rem 0 0 0; color: #4facfe; font-weight: 600;">
                                Predicted Rating: {score:.1f}/5.0 ⭐
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("🤔 No recommendations found based on similar users. Try the Content-Based approach!")

        elif clean_method == 'Hybrid Approach':
            if selected_user not in user_similarity_df.index:
                st.warning("⚠️ User ID not found. Showing content-based recommendations only.")
                idx = books[books['Title'] == selected_book].index[0]
                similar_indices = content_similarity[idx].argsort()[::-1][1:4]
                
                for i, book_idx in enumerate(similar_indices, 1):
                    book_title = books.iloc[book_idx]['Title']
                    book_author = books.iloc[book_idx]['Author']
                    book_genre = books.iloc[book_idx]['Genre']
                    
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">#{i} {book_title}</h4>
                        <p style="margin: 0; color: #a0a0a0; font-size: 0.9rem;">by {book_author} • {book_genre}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                content_idx = books[books['Title'] == selected_book].index[0]
                content_scores = pd.Series(content_similarity[content_idx], index=books['Title'])

                sim_users = user_similarity_df[selected_user].sort_values(ascending=False)[1:]
                user_ratings = ratings_matrix.loc[selected_user]
                unrated_books = user_ratings[user_ratings == 0].index

                user_scores = {}
                for book in unrated_books:
                    book_ratings = ratings_matrix[book]
                    sim_scores = sim_users[book_ratings > 0]
                    ratings_vals = book_ratings[book_ratings > 0]
                    if not sim_scores.empty:
                        score = np.dot(sim_scores, ratings_vals) / sim_scores.sum()
                        user_scores[book] = score

                hybrid_scores = pd.Series(dtype=float)
                for book in books['Title']:
                    c_score = content_scores.get(book, 0)
                    u_score = user_scores.get(book, np.nan)
                    if np.isnan(u_score):
                        hybrid_scores[book] = c_score
                    else:
                        hybrid_scores[book] = (c_score + u_score) / 2

                hybrid_scores = hybrid_scores.drop(selected_book)
                top3 = hybrid_scores.sort_values(ascending=False).head(3)
                
                st.markdown("*Using our advanced hybrid algorithm combining content and user preferences:*")
                
                for i, (book, score) in enumerate(top3.items(), 1):
                    book_info = books[books['Title'] == book].iloc[0]
                    
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4 style="margin: 0 0 0.5rem 0; color: #764ba2;">#{i} {book}</h4>
                        <p style="margin: 0; color: #a0a0a0; font-size: 0.9rem;">
                            by {book_info['Author']} • {book_info['Genre']}
                        </p>
                        <p style="margin: 0.5rem 0 0 0; color: #00f2fe; font-weight: 600;">
                            Hybrid Score: {score:.2f} 🎯
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        # Add a success message
        st.markdown("""
        <div class="success-message">
            <strong>🎉 Recommendations Ready!</strong> Happy reading! Don't forget to rate these books after you read them.
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #a0a0a0; font-size: 0.9rem;'>"
    "Made with ❤️ using advanced machine learning algorithms • "
    "Discover your next favorite book today!"
    "</p>", 
    unsafe_allow_html=True
)