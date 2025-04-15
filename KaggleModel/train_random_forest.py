from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluate_performance import evaluatePerformance

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "cc_data.csv")
cc_df = pd.read_csv(file_path)

# Split data to isolate target variable
X = cc_df.drop(columns=['fraud'], axis=1)
y = cc_df['fraud']

# Create a LabelEncoder object
le = LabelEncoder()

# Transform categorical data
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col])

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
# Train the model
random_forest_model = RandomForestClassifier(criterion='entropy', random_state=42)
random_forest_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(random_forest_model, 'random_forest_model.plk')

# Predict target variable
y_test = random_forest_model.predict(X_test)
evaluatePerformance(random_forest_model, X_test, y_test)