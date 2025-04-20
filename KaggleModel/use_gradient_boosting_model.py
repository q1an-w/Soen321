import numpy as np
import os
import joblib

# Load model and scaler
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "fraud_model_gb.pkl")  # Load the gradient boosting model
scaler_path = os.path.join(script_dir, "gb_scaler.pkl")

gb_model = joblib.load(model_path)  # Load the GradientBoostingClassifier model
gb_scaler = joblib.load(scaler_path)  # Load the scaler

# Function to predict fraud probability
def predict_fraud_v3(transaction):
    transaction = np.array(transaction).reshape(1, -1)
    transaction = gb_scaler.transform(transaction)
    fraud_probability = gb_model.predict_proba(transaction)[:, 1]
    return round(fraud_probability[0] * 100, 2)

# Example scenarios
example_ambiguous = [5.0, 0.5, 1.2, 1.0, 0.0, 1.0, 1.0]  # Example of ambiguous transaction
example_fraudulent = [3.0, 1.5, 5.0, 1.0, 1.0, 0.0, 1.0]  # Example of fraudulent transaction
example_real = [10.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0]  # Example of real (non-fraudulent) transaction

# Calculate fraud probability for each scenario
fraud_prob_ambiguous = predict_fraud_v3(example_ambiguous)
fraud_prob_fraudulent = predict_fraud_v3(example_fraudulent)
fraud_prob_real = predict_fraud_v3(example_real)

# Print results
print(f"Ambiguous Transaction - Fraud Probability: {fraud_prob_ambiguous}%")
print(f"Fraudulent Transaction - Fraud Probability: {fraud_prob_fraudulent}%")
print(f"Real Transaction - Fraud Probability: {fraud_prob_real}%")
