import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ###########################################################################################
## Poisoining attack
## In a white-box adversarial attack, we assume the attacker has access to the training data
# ###########################################################################################

## Retrieve model
model_path = input("Please enter the path to the model file: ")
model = joblib.load(model_path)

## Retrieve training data
csv_path = input("Please enter the path to the csv file: ")
data = pd.read_csv(csv_path)
X = data.drop(columns=['fraud'])
y = data["fraud"].values  # Labels (0 = Not Fraud, 1 = Fraud)

# Extract feature importances
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)

# Modify columns of training data with the most feature importance
data_copy = data.copy()
data_copy['ratio_to_median_purchase'] = data_copy['ratio_to_median_purchase'] * 2
data_copy['online_order'] = 0
data_copy['distance_from_home'] = 5


