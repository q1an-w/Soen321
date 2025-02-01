import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib  # For loading the scaler

# Load model and scaler
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "fraud_model.h5")
scaler_path = os.path.join(script_dir, "scaler.pkl")

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Function to predict fraud probability
def predict_fraud(transaction):
    """
    Predicts fraud probability (0-100%) for a given transaction.
    
    :param transaction: List or numpy array representing transaction features (without Class)
    :return: Fraud probability as a percentage
    """
    transaction = np.array(transaction).reshape(1, -1)
    transaction[:, 1:] = scaler.transform(transaction[:, 1:])  # Standardize features
    transaction = transaction.reshape(1, transaction.shape[1], 1)  # Reshape for CNN
    fraud_probability = model.predict(transaction)[0][0]
    return round(fraud_probability * 100, 2)

# Example usage
example_transaction = np.random.rand(1, 30)  # Replace with real transaction data
fraud_prob = predict_fraud(example_transaction)
print(f"Fraud Probability: {fraud_prob}%")
