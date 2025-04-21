import joblib

# Load existing model that gives the _loss error
old_model = joblib.load("fraud_model_gb.pkl")

# Re-save model that has no legacy metadata
joblib.dump(old_model, "fraud_model_gb_clean.pkl")
