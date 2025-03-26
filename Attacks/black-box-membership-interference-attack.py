import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score

# In a black-box membership interference attack, we assume that the attacker only has access to the outputs of the model.

# Get model
model_path = os.path.join(os.path.dirname(__file__), '..', 'KaggleModel', 'fraud_model_dt.pkl')
target_model = joblib.load(model_path)

# Get data from black box adversarial attack
csv_path = os.path.join(os.path.dirname(__file__), 'output_file.csv')
data = pd.read_csv(csv_path)
data = pd.read_csv(csv_path)
X = data.drop(columns=['target']).values
y = data["target"].values  # Labels (0 = Not Fraud, 1 = Fraud)

# Train shadow model
# Used to generate dataset to train attack model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
shadow_model = DecisionTreeClassifier(random_state=42)
shadow_model.fit(X_train, y_train)

# Membership labels: 1 if in train set, 0 if in test set
train_conf = target_model.predict_proba(X_train)
test_conf = target_model.predict_proba(X_test)

X_attack = np.vstack((train_conf, test_conf))
y_attack = np.hstack((np.ones(len(train_conf)), np.zeros(len(test_conf))))

# Split attack dataset
X_attack_train, X_attack_test, y_attack_train, y_attack_test = train_test_split(X_attack, y_attack, test_size=0.2, random_state=42)

# Train attack model
attack_model = DecisionTreeClassifier(random_state=42)
attack_model.fit(X_attack_train, y_attack_train)

# Evaluate attack success
y_pred = attack_model.predict(X_attack_test)
attack_acc = accuracy_score(y_attack_test, y_pred)

print("Attack Model Accuracy:", attack_acc)