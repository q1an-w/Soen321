import numpy as np
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from art.attacks.inference.membership_inference import (
    MembershipInferenceBlackBox,
    ShadowModels,
)
from art.estimators.classification.scikitlearn import (
    ScikitlearnRandomForestClassifier,
    ScikitlearnDecisionTreeClassifier,
)

# In a black-box membership interference attack, we assume that the attacker has access to the outputs of the model.

# Get scaler
scaler_path = os.path.join(os.path.dirname(__file__), "..", "KaggleModel", "scaler.pkl")
scaler = joblib.load(scaler_path)

# Get target model
model_path = os.path.join(
    os.path.dirname(__file__), "..", "KaggleModel", "fraud_model_dt.pkl"
)
target_model = joblib.load(model_path)


def random_record_callback():
    """Callback function to generate random records."""
    out = np.random.rand(7) * 10
    out = np.array(out).reshape(1, -1)
    out = scaler.transform(out).flatten()
    return out


def randomize_features_callback(record: np.ndarray, num_features: int):
    """Callback function to randomize features."""
    new_record = record.copy()
    for _ in range(num_features):
        new_val = np.random.rand(7) * 0
        new_val = np.array(new_val).reshape(1, -1)
        new_val = scaler.transform(new_val).flatten()
        new_record[np.random.randint(0, 7)] = new_val[np.random.randint(0, 7)]
    return new_record


# Create shadow model
art_shadow_model = ScikitlearnRandomForestClassifier(
    model=RandomForestClassifier(random_state=42, max_depth=5)
)
art_shadow_model.set_params(nb_classes=2)
art_shadow_model = ShadowModels(
    shadow_model_template=art_shadow_model, num_shadow_models=2, random_state=42
)

# Generate synthetic data
art_target_model = ScikitlearnDecisionTreeClassifier(model=target_model)
member_data, nonmember_data = art_shadow_model.generate_synthetic_shadow_dataset(
    target_classifier=art_target_model,
    dataset_size=10000,
    max_features_randomized=7,
    random_record_fn=random_record_callback,
    randomize_features_fn=randomize_features_callback,
    max_retries=100,
)

# Split data
attack_features = np.vstack((member_data[0], nonmember_data[0]))
attack_labels = np.concatenate(
    (np.ones(len(member_data[0])), np.zeros(len(nonmember_data[0])))
)
X_train, X_test, y_train, y_test = train_test_split(
    attack_features, attack_labels, test_size=0.2, random_state=42
)

# # Train attack model
attack = MembershipInferenceBlackBox(estimator=art_target_model, attack_model_type="dt")
attack.fit(X_train, y_train, X_test, y_test)

# Evaluate attack performance
csv_path = os.path.join(os.path.dirname(__file__), "..", "KaggleModel", "cc_data.csv")
data = pd.read_csv(csv_path)
member_data = data.drop(columns=["fraud"]).values
member_data = scaler.transform(member_data)
nonmember_data = np.random.rand(10000, 7) * 1000
nonmember_data = scaler.transform(nonmember_data)

member_preds = attack.infer(member_data, np.ones(len(member_data)))
nonmember_preds = attack.infer(nonmember_data, np.zeros(len(nonmember_data)))

preds = np.concatenate((member_preds, nonmember_preds))
true_labels = np.concatenate((np.ones(len(member_data)), np.zeros(len(nonmember_data))))

correct_predictions = 0
for i in range(len(preds)):
    if preds[i] == true_labels[i]:
        correct_predictions += 1
accuracy = correct_predictions / len(preds)
print("Accuracy: ", accuracy)
