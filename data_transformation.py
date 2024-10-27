import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

class DataTransformation:
    def __init__(self, ratings: pd.DataFrame):
        self.ratings = ratings

    def transform(self):
        # Filter and reduce the dataset size
        active_users = self.ratings['userId'].value_counts()
        active_users = active_users[active_users >= 50].index
        filtered_ratings = self.ratings[self.ratings['userId'].isin(active_users)]

        popular_movies = filtered_ratings['movieId'].value_counts()
        popular_movies = popular_movies[popular_movies >= 100].index
        filtered_ratings = filtered_ratings[filtered_ratings['movieId'].isin(popular_movies)]

        # Create user-movie sparse matrix
        user_movie_matrix = csr_matrix(
            (filtered_ratings['rating'], (filtered_ratings['userId'], filtered_ratings['movieId']))
        )

        # Prepare features and labels
        X = filtered_ratings[['userId', 'movieId']]
        y = filtered_ratings['rating']
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test, user_movie_matrix
