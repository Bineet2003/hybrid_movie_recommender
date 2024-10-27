from flask import Flask, request, jsonify
from data_ingestion import load_data
from data_transformation import preprocess_data
from model_trainer import train_model, make_prediction
import joblib

app = Flask(__name__)

# Load and preprocess data
data_path = "data.csv"  # Change to your data file location
data = load_data(data_path)
preprocessed_data, preprocessor = preprocess_data(data)

# Train the model
model = train_model(preprocessed_data)

# Save the model
joblib.dump(model, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

@app.route('/')
def home():
    return "Welcome to the Hybrid Movie Recommender!"

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions based on user input"""
    input_data = request.get_json()  # Expecting JSON input
    user_id = input_data.get('userId')
    movie_id = input_data.get('movieId')

    if user_id is None or movie_id is None:
        return jsonify({"error": "Please provide userId and movieId"}), 400

    # Make prediction
    prediction = make_prediction(model, preprocessor, user_id, movie_id)

    return jsonify({"predicted_rating": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
