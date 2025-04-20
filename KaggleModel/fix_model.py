import joblib

# Load the already-trained model from your friend's file
model = joblib.load("random_forest_model.pkl")

# Re-save the model with a new name (don't overwrite!)
joblib.dump(model, "random_forest_model_fixed.pkl")

print("Model successfully re-exported.")
