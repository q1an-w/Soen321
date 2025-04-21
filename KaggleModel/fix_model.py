import joblib

# Load already trained model
model = joblib.load("random_forest_model.pkl")

# Re-save the model with a new name (don't overwrite!)
joblib.dump(model, "random_forest_model_fixed.pkl")

print("Model successfully re-exported.")
