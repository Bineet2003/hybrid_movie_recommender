from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the models and the movies dataset
knn_model = joblib.load('knn_model.pkl')
movies = pd.read_csv('movies.csv')

app = Flask(__name__)

# Create a TF-IDF Vectorizer for movie genres
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Function to get content-based recommendations
def get_content_based_recommendations(movie_id, num_recommendations=5):
    cosine_sim = cosine_similarity(tfidf_matrix[movies.index[movies['movieId'] == movie_id]], tfidf_matrix)
    similar_indices = cosine_sim[0].argsort()[-num_recommendations-1:-1][::-1]
    return movies['movieId'].iloc[similar_indices].tolist()

# Hybrid recommendation function with combined scoring
def hybrid_recommendations(user_id, movie_id, num_recommendations=5):
    # Get content-based recommendations
    content_recommendations = get_content_based_recommendations(movie_id, num_recommendations)

    # Get user-based collaborative filtering recommendations (example using KNN model)
    user_recommendations = knn_model.kneighbors([[user_id]], num_neighbors=5, return_distance=False)
    
    # Combine recommendations
    combined_scores = {}

    for movie in content_recommendations:
        combined_scores[movie] = combined_scores.get(movie, 0) + 1  # Content score

    for movie in user_recommendations[0]:
        combined_scores[movie] = combined_scores.get(movie, 0) * 1.5  # Collaborative score

    # Sort movies by combined score and get top recommendations
    top_movies = sorted(combined_scores, key=combined_scores.get, reverse=True)[:num_recommendations]
    return top_movies

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form.get('userId', type=int)
    movie_id = request.form.get('movieId', type=int)

    # Get combined recommendations using the new hybrid function
    combined_recommendations = hybrid_recommendations(user_id, movie_id, num_recommendations=5)

    # Convert recommended movie IDs to titles
    recommended_movies = movies[movies['movieId'].isin(combined_recommendations)]

    return render_template('index.html', recommendations=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
