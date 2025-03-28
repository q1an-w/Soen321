import numpy as np
import joblib
import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KaggleModel.use_model import predict_fraud
from evaluate_performance import evaluatePerformance

# In a black-box membership interference attack, we assume that the attacker has access to the outputs of the model.

# Get scaler
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'KaggleModel', "scaler.pkl")
scaler = joblib.load(scaler_path)

# Get target model
model_path = os.path.join(os.path.dirname(__file__), '..', 'KaggleModel', 'fraud_model_dt.pkl')
target_model = joblib.load(model_path)

# Generate dataset to train shadow model
synthetic_data = np.random.rand(10000, 7)*25
synthetic_data = scaler.transform(synthetic_data)
y = target_model.predict(synthetic_data)
X_train, X_test, y_train, y_test = train_test_split(synthetic_data, y, test_size=0.5, random_state=42)

# Train shadow model
shadow_model = RandomForestClassifier(random_state=42)
shadow_model.fit(synthetic_data, y)

# Get predictions from shadow model
test_preds = shadow_model.predict_proba(X_test)[:, 1]
train_preds = shadow_model.predict_proba(X_train)[:, 1]

# Create labels
test_labels = np.ones(len(test_preds))
train_labels = np.zeros(len(train_preds))

# Train attack model
preds = np.concatenate((train_preds, test_preds)).reshape(-1, 1)
labels = np.concatenate((train_labels, test_labels))
X_train_attack, X_test_attack, y_train_attack, y_test_attack = train_test_split(
    preds, labels, test_size=0.2, random_state=42
)
attack_model = LogisticRegression(random_state=42)
attack_model.fit(X_train_attack, y_train_attack)

evaluatePerformance(attack_model, X_test_attack, y_test_attack)
