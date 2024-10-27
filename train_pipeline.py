from data_ingestion import DataIngestion
from data_transformation import DataTransformation
from model_trainer import ModelTrainer
from logger import Logger

def main():
    logger = Logger()
    logger.log_info("Starting the training pipeline.")
    
    # Data Ingestion
    data_ingestion = DataIngestion('path/to/ratings.csv', 'path/to/movies.csv')
    ratings, movies = data_ingestion.load_data()
    
    # Data Transformation
    data_transformation = DataTransformation(ratings)
    X_train, X_test, y_train, y_test, user_movie_matrix = data_transformation.transform()

    # Model Training
    model_trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    model_trainer.train_models()
    
    logger.log_info("Training pipeline completed.")

if __name__ == "__main__":
    main()
