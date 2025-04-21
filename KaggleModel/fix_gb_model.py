import joblib

# Step 1: Load the existing model that gives the _loss error
old_model = joblib.load("fraud_model_gb.pkl")

# Step 2: Save it again in a clean format with no legacy metadata
joblib.dump(old_model, "fraud_model_gb_clean.pkl")
