import os
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from art.attacks.inference.membership_inference import ShadowModels
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier, ScikitlearnDecisionTreeClassifier

# Load target model
model_path = os.path.join(
    os.path.dirname(__file__), "..", "KaggleModel", "fraud_model_dt.pkl"
)
target_model = joblib.load(model_path)
art_target_model = ScikitlearnDecisionTreeClassifier(model=target_model)

# Load scaler
scaler_path = os.path.join(os.path.dirname(__file__), "..", "KaggleModel", "scaler.pkl")
scaler = joblib.load(scaler_path)

# Callbacks
def random_record_callback():
    """Callback function to generate random records."""
    out = np.random.rand(7) * 5
    out = np.array(out).reshape(1, -1)
    out = scaler.transform(out).flatten()
    return out


def randomize_features_callback(record: np.ndarray, num_features: int):
    """Callback function to randomize features."""
    new_record = record.copy()
    for _ in range(num_features):
        new_val = np.random.rand(7) * 5
        new_val = np.array(new_val).reshape(1, -1)
        new_val = scaler.transform(new_val).flatten()
        new_record[np.random.randint(0, 7)] = new_val[np.random.randint(0, 7)]
    return new_record


# Create shadow models
art_shadow_model = ScikitlearnRandomForestClassifier(
    model=RandomForestClassifier(random_state=42, max_depth=5)
)
art_shadow_model.set_params(nb_classes=2)
art_shadow_model = ShadowModels(
    shadow_model_template=art_shadow_model, num_shadow_models=2, random_state=42
)

# Generate member and non-member data
member_data, nonmember_data = art_shadow_model.generate_synthetic_shadow_dataset(
    target_classifier=art_target_model,
    dataset_size=10000,
    max_features_randomized=7,
    random_record_fn=random_record_callback,
    randomize_features_fn=randomize_features_callback,
    max_retries=100,
)

# Combine member and non-member data
synthetic_data = np.vstack((member_data[0], nonmember_data[0]))

# Get the leaf indices for training samples
train_leaf_indices = target_model.apply(synthetic_data)

# Perform membership inference attack
csv_path = os.path.join(os.path.dirname(__file__), "..", "KaggleModel", "cc_data.csv")
data = pd.read_csv(csv_path)
member_data = data.drop(columns=["fraud"]).values
member_data = scaler.transform(member_data)
target_leaf_indices = target_model.apply(member_data)

# Check if target samples share leaf nodes with training samples
membership_predictions = [leaf in train_leaf_indices for leaf in target_leaf_indices]

# Evaluate performance
count = 0
for is_member in membership_predictions:
    if is_member:
        count += 1
accuracy = count / len(member_data)
print('Accuracy: ', accuracy)
print('Pred: ', count, '\tActual: ', len(member_data))