import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluate_performance import evaluatePerformance

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "cc_data.csv")
df = pd.read_csv(file_path)

# Drop missing values if any
if df.isnull().sum().sum() > 0:
    df = df.dropna()

# Split features and target
X = df.drop(columns=['fraud']).values
y = df['fraud'].values

# Standardize features
gb_scaler = StandardScaler()
X = gb_scaler.fit_transform(X)

# Save scaler
joblib.dump(gb_scaler, os.path.join(script_dir, "gb_scaler.pkl"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Evaluate model
print(f"Gradient Boosting Model Performance Metrics:\n")
evaluatePerformance(gb_model, X_test, y_test)

# Save the trained model
joblib.dump(gb_model, os.path.join(script_dir, "fraud_model_gb.pkl"))
print("Model saved as fraud_model_gb.pkl")