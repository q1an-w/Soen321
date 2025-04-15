import numpy as np
import os
import joblib  # For loading the model and scaler

# Load model and scaler
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "random_forest_model.pkl")  # Load the random forest model
scaler_path = os.path.join(script_dir, "scaler.pkl")

model = joblib.load(model_path)  # Load the RandomForestClassifier model
scaler = joblib.load(scaler_path)  # Load the scaler

# Function to predict fraud probability
def predict_fraud_v2(transaction):
    """
    Predicts fraud probability (0-100%) for a given transaction.
    
    :param transaction: List or numpy array representing transaction features (without Class)
    :return: Fraud probability as a percentage
    """
    transaction = np.array(transaction).reshape(1, -1)
    transaction = scaler.transform(transaction)  # Standardize features
    fraud_probability = model.predict_proba(transaction)[:, 1]  # Get probability of the positive class (fraud)
    return round(fraud_probability[0] * 100, 2)

# Example scenarios
example_ambiguous = [5.0, 0.5, 1.2, 1.0, 0.0, 1.0, 1.0]  # Example of ambiguous transaction
example_fraudulent = [3.0, 1.5, 5.0, 1.0, 1.0, 0.0, 1.0]  # Example of fraudulent transaction
example_real = [10.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0]  # Example of real (non-fraudulent) transaction

# Calculate fraud probability for each scenario
fraud_prob_ambiguous = predict_fraud_v2(example_ambiguous)
fraud_prob_fraudulent = predict_fraud_v2(example_fraudulent)
fraud_prob_real = predict_fraud_v2(example_real)

# Print results
print(f"Ambiguous Transaction - Fraud Probability: {fraud_prob_ambiguous}%")
print(f"Fraudulent Transaction - Fraud Probability: {fraud_prob_fraudulent}%")
print(f"Real Transaction - Fraud Probability: {fraud_prob_real}%")
