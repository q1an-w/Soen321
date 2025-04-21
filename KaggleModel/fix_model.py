import joblib

# Load already trained model
model = joblib.load("random_forest_model.pkl")

# Re-save new model with a new name
joblib.dump(model, "random_forest_model_fixed.pkl")

print("Model successfully re-exported.")
