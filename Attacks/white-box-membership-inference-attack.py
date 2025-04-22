import numpy as np
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import pandas as pd

# Get scaler
scaler_path = os.path.join(os.path.dirname(__file__), "..", "KaggleModel", "scaler.pkl")
scaler = joblib.load(scaler_path)

model_path = os.path.join(
    os.path.dirname(__file__), "..", "KaggleModel", "fraud_model_dt.pkl"
)
target_model = joblib.load(model_path)

def extract_features(model, X):
    # Leaf indices for each tree
    leaf_indices = model.apply(X)

    # Predicted probabilities
    pred_probs = model.predict_proba(X)

    return np.hstack([leaf_indices.reshape(-1,1), pred_probs])

def eval_attack(attack_model):
    # Evaluate attack performance
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "KaggleModel", "cc_data.csv"
    )
    data = pd.read_csv(csv_path)
    member_data = data.sample(n=100000)
    member_data = member_data.drop(columns=["fraud"]).values
    member_data = scaler.transform(member_data)
    member_data = extract_features(target_model, member_data)

    nonmember_data = np.random.rand(100000, 7) * 10000
    nonmember_data = scaler.transform(nonmember_data)
    nonmember_data = extract_features(target_model, nonmember_data)

    member_pred = attack_model.predict(member_data)
    nonmember_pred = attack_model.predict(nonmember_data)

    member_accuracy = np.sum(member_pred) / len(member_data)
    nonmember_accuracy = np.sum(nonmember_pred) / len(nonmember_data)

    print("Attack Member Acc:", member_accuracy)
    print("Attack Non-Member Acc:", nonmember_accuracy)

data_path = os.path.join(os.path.dirname(__file__), "dt_shadow_dataset.pkl")
shadow_dataset = joblib.load(data_path)
(member_x, member_y, member_predictions), (
    nonmember_x,
    nonmember_y,
    nonmember_predictions,
) = shadow_dataset

# Get member and non-member features
X_member_features = extract_features(target_model, member_x)
X_nonmember_features = extract_features(target_model, nonmember_x)

# Label 1 for members and 0 for non-members
X_attack = np.vstack([X_member_features, X_nonmember_features])
y_attack = np.array([1] * len(member_x) + [0] * len(nonmember_x))

attack_dt = DecisionTreeClassifier(random_state=42)
attack_dt.fit(X_attack, y_attack)
attack_gb = GradientBoostingClassifier(random_state=42)
attack_gb.fit(X_attack, y_attack)
attack_rf = RandomForestClassifier(random_state=42)
attack_rf.fit(X_attack, y_attack)

print('Decision Tree')
eval_attack(attack_dt)
print('Gradient Boosting')
eval_attack(attack_gb)
print('Random Forest')
eval_attack(attack_rf)