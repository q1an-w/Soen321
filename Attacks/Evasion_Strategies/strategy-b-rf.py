import pandas as pd
import numpy as np
import joblib

# Load data generated from surrogate Random Forest model
df = pd.read_csv("../rf_output_file.csv")
frauds = df[df["target"] == 1].copy()
X_original = frauds.drop(columns=["target"]).copy()

# Strategy B logic: increase feature3 and feature6
def strategy_b(row):
    row["feature3"] += 1
    row["feature6"] += 1
    return row

# Apply the attack
X_mod = X_original.copy().apply(strategy_b, axis=1)

# Load model and scaler
model = joblib.load("../../KaggleModel/random_forest_model.pkl")
scaler = joblib.load("../../KaggleModel/rf_scaler.pkl")
X_scaled = scaler.transform(X_mod.values)  # using .values to avoid feature name mismatch warning

# Predict and evaluate
predictions = model.predict(X_scaled)
frauds["Prediction"] = predictions
frauds["Evasion_Success"] = predictions == 0
frauds.to_csv("results_strategy_b_rf.csv", index=False)

success = np.sum(predictions == 0)
total = len(predictions)
print(f"[RF] Strategy B: {success}/{total} successful evasions ({(success/total)*100:.2f}%)")
