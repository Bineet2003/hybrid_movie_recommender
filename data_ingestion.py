import pandas as pd

class DataIngestion:
    def __init__(self, ratings_path: str, movies_path: str):
        self.ratings_path = ratings_path
        self.movies_path = movies_path

    def load_data(self):
        ratings = pd.read_csv(self.ratings_path)
        movies = pd.read_csv(self.movies_path)
        return ratings, movies
