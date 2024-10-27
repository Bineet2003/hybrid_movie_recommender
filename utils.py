import pandas as pd
import joblib

def load_pickle(file_path: str):
    """Load a pickled object from a file."""
    return joblib.load(file_path)

def save_pickle(data, file_path: str):
    """Save an object to a pickle file."""
    joblib.dump(data, file_path)

def preprocess_ratings_data(data: pd.DataFrame):
    """Preprocess ratings data."""
    # Your preprocessing logic here (if any)
    return data
