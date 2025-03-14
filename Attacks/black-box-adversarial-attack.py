import random
import numpy as np
import pandas as pd
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KaggleModel.use_model import predict_fraud
from evaluate_performance import evaluatePerformance

# ##################################################################################################################################
## Surrogate Model
## In a black-box adversarial attack, we assume the attacker only has access to the input/output of the model like any other user
# ##################################################################################################################################

# Generate random inputs
def generate_random_input():
    input = [random.randint(0, 5) for x in range(7)] # achieved through trial and error (found that range from 0 to 5 had better results)
    return input

# Path to csv file
csv_path = os.path.join(os.path.dirname(__file__), 'output_file.csv')

# Create text file -- if you want to create the file again, uncomment this out but this file was found to be very efficient through trial and error
# with open(csv_path, 'w') as file:
#     file.write('feature1,feature2,feature3,feature4,feature5,feature6,feature7,target\n')

#     for x in range(10000):
#         attempt = generate_random_input()
#         file.write(",".join(map(str, attempt)))
#         file.write(",")
#         result = predict_fraud(attempt)
#         if result == 100.0:
#             result = 1
#         else:
#             result = 0
#         file.write(str(result))
#         file.write("\n")

# Training surrogate model
data = pd.read_csv(csv_path)
X = data.drop(columns=['target'])
y = data["target"].values  # Labels (0 = Not Fraud, 1 = Fraud)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate surrogate model
evaluatePerformance(model,X_test,y_test)