import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("../output_file.csv")
frauds = df[df["target"] == 1].copy()
X_original = frauds.drop(columns=["target"]).copy()

def strategy_a(row):
    if row["feature5"] > 0:
        row["feature5"] -= 1
    if row["feature7"] > 0:
        row["feature7"] -= 1
    return row

X_mod = X_original.copy().apply(strategy_a, axis=1)

model = joblib.load("../../KaggleModel/fraud_model_dt.pkl")
scaler = joblib.load("../../KaggleModel/scaler.pkl")
X_scaled = scaler.transform(X_mod)
predictions = model.predict(X_scaled)

frauds["Prediction"] = predictions
frauds["Evasion_Success"] = predictions == 0
frauds.to_csv("results_strategy_a.csv", index=False)


success = np.sum(predictions == 0)
total = len(predictions)
print(f"Strategy A: {success}/{total} successful evasions ({(success/total)*100:.2f}%)")
