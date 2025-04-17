from flask import Flask, request, jsonify
import numpy as np
import os
import joblib
import time
import csv
import datetime
import threading
import psutil

api = Flask(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, "fraud_model_dt.pkl"))
scaler = joblib.load(os.path.join(os.path.join(script_dir, "scaler.pkl")))

def log_to_csv(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

def log_system_metrics():
    while True:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cpu_usage = psutil.cpu_percent(interval=1)  # CPU usage in percent
        memory_usage = psutil.virtual_memory().percent  # Memory usage in percent
        
        # Ethernet throughput estimate
        net_io = psutil.net_io_counters()
        bytes_sent = net_io.bytes_sent
        bytes_recv = net_io.bytes_recv
        time.sleep(1)  # Wait 1 second to measure difference
        net_io_after = psutil.net_io_counters()
        bytes_sent_after = net_io_after.bytes_sent
        bytes_recv_after = net_io_after.bytes_recv
        # Mbps (bits per second / 1000000)
        sent_diff = (bytes_sent_after - bytes_sent) * 8 / 1_000_000
        recv_diff = (bytes_recv_after - bytes_recv) * 8 / 1_000_000
        throughput = sent_diff + recv_diff
        
        log_to_csv('defender_metrics.csv', [current_time, cpu_usage, memory_usage, throughput])

with open('defender_metrics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['date', 'cpu_usage', 'memory_usage', 'ethernet_throughput'])

threading.Thread(target=log_system_metrics, daemon=True).start()

@api.route("/predict", methods=["POST"])
def predict_fraud():
    start = time.time()
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
    end = time.time()

    processing_time = end - start
    print(f"Request processed in {processing_time:.4f} seconds")

    return jsonify({"fraud_probability": round(fraud_prob, 2), "processing_time": processing_time})

if __name__ == "__main__":
    api.run(host="0.0.0.0", port=5000)