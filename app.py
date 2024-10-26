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

# Function to get content-based recommendations using cosine similarity
def get_content_based_recommendations(movie_id):
    # Compute cosine similarity for the given movie
    cosine_sim = cosine_similarity(tfidf_matrix[movies.index[movies['movieId'] == movie_id]], tfidf_matrix)
    # Get the index of similar movies
    similar_indices = cosine_sim[0].argsort()[::-1]
    # Return sorted movie indices (excluding the first one which is the same movie)
    return [(movies['movieId'].iloc[i], cosine_sim[0][i]) for i in similar_indices if i != movies.index[movies['movieId'] == movie_id][0]]

# Hybrid recommendation function with combined scoring (multiplying scores)
def hybrid_recommendations(user_id, movie_id, num_recommendations=5):
    # Get content-based recommendations
    content_recommendations = get_content_based_recommendations(movie_id)

    # Get user-based collaborative filtering recommendations
    user_recommendations = knn_model.kneighbors([[user_id]], n_neighbors=num_recommendations, return_distance=False)[0]

    # Combine recommendations
    combined_scores = {}

    # Store content-based scores
    for movie_id, score in content_recommendations:
        combined_scores[movie_id] = score  # Content score

    # Multiply scores from collaborative filtering
    for movie_id in user_recommendations:
        if movie_id in combined_scores:
            combined_scores[movie_id] *= 1.5  # Adjust weight as needed

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
