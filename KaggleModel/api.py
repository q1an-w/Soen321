from flask import Flask, request, jsonify
import numpy as np
import os
import joblib

api = Flask(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, "fraud_model_dt.pkl"))
scaler = joblib.load(os.path.join(os.path.join(script_dir, "scaler.pkl")))

@api.route("/predict", methods=["POST"])
def predict_fraud():
    data = request.json
    transaction = [
        data["distance_from_home"],
        data["distance_from_last_transaction"],
        data["ratio_to_median_purchase_price"],
        data["repeat_retailer"],
        data["used_chip"],
        data["used_pin_number"],
        data["online_order"]
    ]
    transaction = np.array(transaction).reshape(1, -1)
    transaction = scaler.transform(transaction)
    fraud_prob = model.predict_proba(transaction)[0, 1] * 100
    return jsonify({"fraud_probability": round(fraud_prob, 2)})

if __name__ == "__main__":
    api.run(host="0.0.0.0", port=5000)