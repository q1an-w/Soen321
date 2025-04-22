# SOEN 321 Project
## Team Members

Karina Sanchez-Duran (ID: 40189860)

Qian Yi Wang (ID: 40211303)

Paul Humennyj (ID: 40209588)

Vanessa DiPietrantonio (ID: 40189938) 

Mohamad Mounir Yassin (ID: 40198854)

Yash Patel (ID: 40175454)

## Brief Project Overview

The goal of this project is to analyze and compare the effects of various attack methods on three different machine learning algorithms with the intention of identifying critical vulnerabilities and propose countermeasures for future improvements.

## File Explanation

### 1) Report folder
   - Inside the report folder, there is a PDF of the final report submission.
  
### 2) Kaggle Model folder

  - Inside the Kaggle Model folder, there are several files that relate to the training/saving of the three different models built in this project: decision tree, random forest and gradient boosting.
  - The files called **train_model.py**, **train_random_forest.py**, **train_gradient_boosting.py** contain the code to train the decision tree, random forest and gradient boosting models, respectively.
  - The train_model.py file created the **fraud_model.plk** and **scaler.plk** files which are the saved model and scaler for the decision tree.
  - The train_random_forest.py file created the r**andom_forest_model.plk** and **rf_scaler.plk** files which are the saved model and scaler for the random forest.
  - The train_gradient_boosting.py file created the **fraud_model_gb.plk** and **gb_scaler.plk** files which are the saved model and scaler for the gradient boosting.
  - The files called **use_model.py**, **use_random_forest_model.py** and **use_gradient_boosting_model.py** files contain the code that use the saved model/scalar to classify credit card transactions for the decision tree, random forest and gradient boosting, respectively. 
  
### 3) Attacks folder

  - The **Evasion_Strategies** folder contains the Python scripts and output files for the three different evasion attack strategies implemented on the three different models. All csv files are the outputted results of the attack and all .py files contain the code for the attack.
  - The **black-box-adversarial-attack.py** script contains the code that generates a surrogate model entirely from scratch for all three different models. This attack generates three csv files that are used to train the surrogate models:
      - **output_file.csv** (for decision tree surrogate training data)
      - **rf_output_file.csv** (for random forest surrogate training data)
      - **gb_output_file.csv** (for gradient boosting surrogate training data)
  - The **black-box-membership-inference.py** script contains the code that generates a black-box membership inference attack on all three models. This file also creates 2 training datasets for the attack model:
      - **dt_shadow_dataset.pkl** for the decision tree and random forest model
      - **gb_shadow_dataset.pkl** for the gradient boosting model
  - The **white-box-membership-inference.py** script contains the code that generates a white-box membership inference attack on all three models.
  - The **poison_attacks.ipynb** file implements a poisoning attack on all three models.
  - The **slowris.py** script implements several DoS attacks.
  - The **DoS_Attacks.xlsx** is an excel sheet that displays the results of the three different DoS attacks.

### 4) evaluate_performance.py

- Python script to evaluate the performance of models (used repeatedly thoughout project)

### 5) .gitignore

- We have placed the training data in gitignore because the csv file is far too large but it can be found on Kaggle at this link:  https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud 
