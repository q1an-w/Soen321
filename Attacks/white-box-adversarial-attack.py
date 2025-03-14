import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluate_performance import evaluatePerformance

# ###########################################################################################
## Poisoning attack
## In a white-box adversarial attack, we assume the attacker has access to the training data
# ###########################################################################################

## Retrieve model
model_path = os.path.join(os.path.dirname(__file__), '..', 'KaggleModel', 'fraud_model_dt.pkl')
model = joblib.load(model_path)

## Retrieve training data
csv_path = os.path.join(os.path.dirname(__file__), '..', 'KaggleModel', 'cc_data.csv')

## Read csv file and extract data
data = pd.read_csv(csv_path)
X = data.drop(columns=['fraud'])
y = data["fraud"].values  # Labels (0 = Not Fraud, 1 = Fraud)

# Extract feature importances -- this part of the code waas run before subsequent parts so the feature importances are known
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)

# Modify columns of training data with the most feature importance
data_copy = data.copy()
data_copy['ratio_to_median_purchase_price'] = 0
data_copy['online_order'] = 1
data_copy['distance_from_home'] = 5
data_copy['used_pin_number'] = 0

# Extract data for poisoned model
new_X = data_copy.drop(columns=['fraud'])
new_y = data_copy["fraud"].values  # Labels (0 = Not Fraud, 1 = Fraud)

# Train-test split with poisoned model
X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.2, random_state=42, stratify=y)
poisoned_model = DecisionTreeClassifier(random_state=42)
poisoned_model.fit(X_train, y_train)

# Evaluate poisoned model
evaluatePerformance(poisoned_model,X_test,y_test)

