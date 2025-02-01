import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib  # For saving the scaler

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "creditcard_2023.csv")
df = pd.read_csv(file_path)

# Check and drop missing values
if df.isnull().sum().sum() > 0:
    df = df.dropna()

# Separate features and labels
X = df.drop(columns=["Class"]).values  # Features
y = df["Class"].values  # Labels (0 = Not Fraud, 1 = Fraud)

# Standardize numerical features
scaler = StandardScaler()
X[:, 1:] = scaler.fit_transform(X[:, 1:])

# Save the scaler for later use
joblib.dump(scaler, os.path.join(script_dir, "scaler.pkl"))

# Reshape for CNN input (CNN expects 3D input: samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
y_pred_proba = model.predict(X_test).flatten()
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"Model AUC-ROC Score: {auc_score:.4f}")

# Save the trained model
model.save(os.path.join(script_dir, "fraud_model.h5"))
print("Model saved as fraud_model.h5")
