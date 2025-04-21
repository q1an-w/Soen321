import pandas as pd
import numpy as np
import joblib

# Load the data and isolate fraudulent cases
df = pd.read_csv("../gb_output_file.csv")
frauds = df[df["target"] == 1].copy()
X_original = frauds.drop(columns=["target"]).copy()

# Strategy C: Add or subtract 1 randomly to each feature
def strategy_c(row):
    for col in row.index:
        row[col] += np.random.choice([-1, 1])
    return row

# Apply evasion strategy
X_mod = X_original.copy().apply(strategy_c, axis=1)

# Load the fixed Gradient Boosting model and scaler
model = joblib.load("../../KaggleModel/fraud_model_gb_clean.pkl")
scaler = joblib.load("../../KaggleModel/gb_scaler.pkl")
X_scaled = scaler.transform(X_mod)

# Predict and compare with target
predictions = model.predict(X_scaled)
frauds["Prediction"] = predictions
frauds["Evasion_Success"] = predictions == 0

# Save results
frauds.to_csv("results_strategy_c_gb.csv", index=False)

# Print success rate
success = np.sum(frauds["Evasion_Success"])
total = len(frauds)
print(f"Strategy C (Gradient Boosting): {success}/{total} successful evasions ({(success/total)*100:.2f}%)")
