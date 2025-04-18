import random
import numpy as np
import pandas as pd
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KaggleModel.use_model import predict_fraud
from KaggleModel.use_random_forest_model import predict_fraud_v2
from KaggleModel.use_gradient_boosting_model import predict_fraud_v3
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
dt_output = os.path.join(os.path.dirname(__file__), 'output_file.csv')
rf_output = os.path.join(os.path.dirname(__file__), 'rf_output_file.csv')
gb_output = os.path.join(os.path.dirname(__file__), 'gb_output_file.csv')

# ############################################## Decision Tree Model Extraction Attack ##############################################

# Creating decision tree output file
if not os.path.exists(dt_output):
    with open(dt_output, 'w') as file:
        file.write('feature1,feature2,feature3,feature4,feature5,feature6,feature7,target\n')

        for x in range(10000):
            attempt = generate_random_input()
            file.write(",".join(map(str, attempt)))
            file.write(",")
            result = predict_fraud(attempt)
            if result == 100.0:
                result = 1
            else:
                result = 0
            file.write(str(result))
            file.write("\n")
else:
    print("Output file for decision tree exists!")

# Reading decision tree output file
dt_data = pd.read_csv(dt_output)
dt_X = dt_data.drop(columns=['target'])
dt_y = dt_data["target"].values
X_train, X_test, y_train, y_test = train_test_split(dt_X, dt_y, test_size=0.2, random_state=42, stratify=dt_y)

# Training decision tree surrogate
print("Training decision tree surrogate....")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("Finished training decision tree surrogate!")

# Evaluating decision tree
print("Evaluating decision tree surrogate....")
evaluatePerformance(dt_model,X_test,y_test)
print("Finished evaluating decision tree surrogate!")

# ############################################## Random Forest Model Extraction Attack ##############################################

# Creating random forest output file
if not os.path.exists(rf_output):
    with open(rf_output, 'w') as file:
        file.write('feature1,feature2,feature3,feature4,feature5,feature6,feature7,target\n')

        for x in range(10000):
            attempt = generate_random_input()
            file.write(",".join(map(str, attempt)))
            file.write(",")
            result = predict_fraud_v2(attempt)
            if result == 100.0:
                result = 1
            else:
                result = 0
            file.write(str(result))
            file.write("\n")
else:
    print("Output file for random forest exists!")

# Reading random forest output file
rf_data = pd.read_csv(rf_output)
rf_X = rf_data.drop(columns=['target'])
rf_y = rf_data["target"].values
X_train, X_test, y_train, y_test = train_test_split(rf_X, rf_y, test_size=0.2, random_state=42, stratify=rf_y)

# Training random forest surrogate
print("Training random forest surrogate....")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("Finished training random forest surrogate!")

# Evaluating random forest
print("Evaluating random forest surrogate....")
evaluatePerformance(rf_model,X_test,y_test)
print("Finished evaluating random forest surrogate!")

# ############################################## Gradient Boosting Model Extraction Attack ##############################################

# Creating gradient boosting output file
if not os.path.exists(gb_output):
    with open(gb_output, 'w') as file:
        file.write('feature1,feature2,feature3,feature4,feature5,feature6,feature7,target\n')

        for x in range(10000):
            attempt = generate_random_input()
            file.write(",".join(map(str, attempt)))
            file.write(",")
            result = predict_fraud_v3(attempt)
            if result > 95.0: # Done as greater than 95% and not 100% because this model predicts with at most a probability of 98% (usually)
                result = 1
            else:
                result = 0
            file.write(str(result))
            file.write("\n")
else:
    print("Output file for gradient boosting exists!")

# Reading gradient boosting output file
gb_data = pd.read_csv(gb_output)
gb_X = gb_data.drop(columns=['target'])
gb_y = gb_data["target"].values
X_train, X_test, y_train, y_test = train_test_split(gb_X, gb_y, test_size=0.2, random_state=42, stratify=gb_y)

# Training gradient boosting surrogate
print("Training gradient booster surrogate....")
rf_model = GradientBoostingClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("Finished training gradient booster surrogate!")

# Evaluating gradient booster
print("Evaluating gradient booster surrogate....")
evaluatePerformance(rf_model,X_test,y_test)
print("Finished evaluating gradient booster surrogate!")