import numpy as np
import joblib
import pandas as pd
import os
from art.attacks.inference.membership_inference import (
    MembershipInferenceBlackBox,
    ShadowModels,
)
from art.estimators.classification.scikitlearn import (
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnRandomForestClassifier,
)

# In a black-box membership interference attack, we assume that the attacker has access to the outputs of the model.

# Get scaler
scaler_path = os.path.join(os.path.dirname(__file__), "..", "KaggleModel", "scaler.pkl")
scaler = joblib.load(scaler_path)


def random_record_callback():
    """Callback function to generate random records."""
    out = np.random.rand(7) * 10000
    out = np.array(out).reshape(1, -1)
    out = scaler.transform(out).flatten()
    return out


def randomize_features_callback(record: np.ndarray, num_features: int):
    """Callback function to randomize features."""
    new_record = record.copy()
    for _ in range(num_features):
        new_val = np.random.rand(7) * 10000
        new_val = np.array(new_val).reshape(1, -1)
        new_val = scaler.transform(new_val).flatten()
        new_record[np.random.randint(0, 7)] = new_val[np.random.randint(0, 7)]
    return new_record


def dt_mia_attack():
    # Get target model
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "KaggleModel", "fraud_model_dt.pkl"
    )
    target_model = joblib.load(model_path)
    art_target_model = ScikitlearnDecisionTreeClassifier(target_model)

    # Generate or get synthetic dataset
    data_path = os.path.join(os.path.dirname(__file__), "dt_shadow_dataset.pkl")
    if os.path.exists(data_path):
        shadow_dataset = joblib.load(data_path)
    else:
        shadow_models = ShadowModels(art_target_model, num_shadow_models=2)
        shadow_dataset = shadow_models.generate_synthetic_shadow_dataset(
            art_target_model,
            10000,
            max_features_randomized=7,
            max_retries=100,
            random_record_fn=random_record_callback,
            randomize_features_fn=randomize_features_callback,
        )
        joblib.dump(shadow_dataset, "dt_shadow_dataset.pkl")

    (member_x, member_y, member_predictions), (
        nonmember_x,
        nonmember_y,
        nonmember_predictions,
    ) = shadow_dataset

    # Create attack model
    attack = MembershipInferenceBlackBox(
        art_target_model, scaler_type="standard", attack_model_type="dt"
    )
    attack.fit(
        member_x,
        member_y,
        nonmember_x,
        nonmember_y,
        member_predictions,
        nonmember_predictions,
    )
    return attack


def gb_mia_attack():
    # Get target model
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "KaggleModel", "fraud_model_gb.pkl"
    )
    target_model = joblib.load(model_path)
    art_target_model = ScikitlearnGradientBoostingClassifier(target_model)

    # Generate or get synthetic dataset
    data_path = os.path.join(os.path.dirname(__file__), "gb_shadow_dataset.pkl")
    if os.path.exists(data_path):
        shadow_dataset = joblib.load(data_path)
    else:
        shadow_models = ShadowModels(art_target_model, num_shadow_models=2)
        shadow_dataset = shadow_models.generate_synthetic_shadow_dataset(
            art_target_model,
            10000,
            max_features_randomized=7,
            max_retries=100,
            random_record_fn=random_record_callback,
            randomize_features_fn=randomize_features_callback,
        )
        joblib.dump(shadow_dataset, "gb_shadow_dataset.pkl")

    (member_x, member_y, member_predictions), (
        nonmember_x,
        nonmember_y,
        nonmember_predictions,
    ) = shadow_dataset

    # Create attack model
    attack = MembershipInferenceBlackBox(
        art_target_model, scaler_type="standard", attack_model_type="gb"
    )
    attack.fit(
        member_x,
        member_y,
        nonmember_x,
        nonmember_y,
        member_predictions,
        nonmember_predictions,
    )
    return attack


def rf_mia_attack():
    # Get target model
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "KaggleModel", "random_forest_model.pkl"
    )
    target_model = joblib.load(model_path)
    art_target_model = ScikitlearnRandomForestClassifier(target_model)

    # Generate or get synthetic dataset
    data_path = os.path.join(os.path.dirname(__file__), "dt_shadow_dataset.pkl")
    if os.path.exists(data_path):
        shadow_dataset = joblib.load(data_path)
    else:
        shadow_models = ShadowModels(art_target_model, num_shadow_models=2)
        shadow_dataset = shadow_models.generate_synthetic_shadow_dataset(
            art_target_model,
            10000,
            max_features_randomized=7,
            max_retries=100,
            random_record_fn=random_record_callback,
            randomize_features_fn=randomize_features_callback,
        )
        joblib.dump(shadow_dataset, "dt_shadow_dataset.pkl")

    (member_x, member_y, member_predictions), (
        nonmember_x,
        nonmember_y,
        nonmember_predictions,
    ) = shadow_dataset

    # Create attack model
    attack = MembershipInferenceBlackBox(
        art_target_model, scaler_type="standard", attack_model_type="rf"
    )
    attack.fit(
        member_x,
        member_y,
        nonmember_x,
        nonmember_y,
        member_predictions,
        nonmember_predictions,
    )
    return attack


def eval_attack(attack_model):
    # Evaluate attack performance
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "KaggleModel", "cc_data.csv"
    )
    data = pd.read_csv(csv_path)
    member_data = data.sample(n=100000)
    member_data = member_data.drop(columns=["fraud"]).values
    member_data = scaler.transform(member_data)
    nonmember_data = np.random.rand(100000, 7) * 10000
    nonmember_data = scaler.transform(nonmember_data)

    member_infer = attack_model.infer(member_data, np.ones(len(member_data)))
    nonmember_infer = attack_model.infer(nonmember_data, np.zeros(len(nonmember_data)))

    member_accuracy = 1 - np.sum(member_infer) / len(member_data)
    nonmember_accuracy = np.sum(nonmember_infer) / len(nonmember_data)

    print("Attack Member Acc:", member_accuracy)
    print("Attack Non-Member Acc:", nonmember_accuracy)


eval_attack(dt_mia_attack())
eval_attack(gb_mia_attack())
eval_attack(rf_mia_attack())
