import numpy as np
import joblib
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import SklearnClassifier

# In a black-box membership interference attack, we assume that the attacker has access to the outputs of the model.

# Get scaler
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'KaggleModel', "scaler.pkl")
scaler = joblib.load(scaler_path)

# Get target model
model_path = os.path.join(os.path.dirname(__file__), '..', 'KaggleModel', 'fraud_model_dt.pkl')
target_model = joblib.load(model_path)

# Generate dataset to train attack model
# synthetic_data = np.random.rand(10000, 7) * 50
# synthetic_data = scaler.transform(synthetic_data)
# np.savetxt("synthetic_data.csv", synthetic_data, delimiter=",")
csv_path = os.path.join(os.path.dirname(__file__), 'synthetic_data.csv')
synthetic_data = np.genfromtxt(csv_path, delimiter=",")
y = target_model.predict(synthetic_data)
X_train, X_test, y_train, y_test = train_test_split(synthetic_data, y, test_size=0.5, random_state=42)

# Train attack model
art_classifier = SklearnClassifier(model=target_model)
attack = MembershipInferenceBlackBox(art_classifier)
attack.fit(X_train, y_train, X_test, y_test)

# Evaluate attack performance
csv_path = os.path.join(os.path.dirname(__file__), '..', 'KaggleModel', 'cc_data.csv')
data = pd.read_csv(csv_path)
X = data.drop(columns=["fraud"]).values
X = scaler.transform(X)
test_data = np.random.rand(10000, 7) * 50
test_data = scaler.transform(test_data)
 
preds = np.concatenate((attack.infer(X, np.ones(len(X))), attack.infer(test_data, np.ones(len(test_data)))))

count = 0
for pred in preds:
    if pred == 0.0:
        count += 1

print("Number of samples not part of the training data out of 1010000: ", count)