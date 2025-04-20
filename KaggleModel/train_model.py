import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import joblib  # For saving the model and scaler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluate_performance import evaluatePerformance

# Load dataset (update this if needed for the file path)
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "cc_data.csv")
df = pd.read_csv(file_path)

# Check and drop missing values
if df.isnull().sum().sum() > 0:
    df = df.dropna()

# Define the column names
columns = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 
           'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order', 'fraud']

# Ensure the DataFrame has the expected column names
df.columns = columns

# Separate features and labels
X = df.drop(columns=["fraud"]).values  # Features
y = df["fraud"].values  # Labels (0 = Not Fraud, 1 = Fraud)

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler for later use
joblib.dump(scaler, os.path.join(script_dir, "scaler.pkl"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"Model AUC-ROC Score: {auc_score:.4f}")
evaluatePerformance(model, X_test, y_test)

# Save the trained model
joblib.dump(model, os.path.join(script_dir, "fraud_model_dt.pkl"))
print("Model saved as fraud_model_dt.pkl")
