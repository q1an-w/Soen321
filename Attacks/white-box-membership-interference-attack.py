import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score

# In a white-box membership interference attack, we assume that the attacker has full access to the model's structure

# Get model
model_path = os.path.join(os.path.dirname(__file__), '..', 'KaggleModel', 'fraud_model_dt.pkl')
target_model = joblib.load(model_path)

# Get data from black box adversarial attack
csv_path = os.path.join(os.path.dirname(__file__), 'output_file.csv')
data = pd.read_csv(csv_path)
data = pd.read_csv(csv_path)
X = data.drop(columns=['target']).values
y = data["target"].values  # Labels (0 = Not Fraud, 1 = Fraud)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Extract leaf indices for training and test samples
train_leaf_indices = target_model.apply(X_train)
test_leaf_indices = target_model.apply(X_test)

# Create attack dataset
X_attack = np.hstack((train_leaf_indices.reshape(-1, 1), target_model.predict_proba(X_train)))
y_attack = np.ones(len(X_train))

X_attack_test = np.hstack((test_leaf_indices.reshape(-1, 1), target_model.predict_proba(X_test)))
y_attack_test = np.zeros(len(X_test))

X_attack_full = np.vstack((X_attack, X_attack_test))
y_attack_full = np.hstack((y_attack, y_attack_test))

# Split attack dataset
X_attack_train, X_attack_val, y_attack_train, y_attack_val = train_test_split(X_attack_full, y_attack_full, test_size=0.2, random_state=42)

# Train attack model
attack_model = DecisionTreeClassifier(random_state=42)
attack_model.fit(X_attack_train, y_attack_train)

# Evaluate attack
y_pred = attack_model.predict(X_attack_val)
attack_acc = accuracy_score(y_attack_val, y_pred)

print("White-Box Attack Accuracy:", attack_acc)