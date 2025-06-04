
# FindDefault (Prediction of Credit Card fraud)

# Problem Statement:

A credit card is one of the most used financial products to make online purchases and payments. Though the Credit cards can be a convenient way to manage your finances, they can also be risky. Credit card fraud is the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash.

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

We have to build a classification model to predict whether a credit card transaction is fraudulent or not.

Credit card transaction dataset contains 'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20','V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount','Class'

# The following steps to solve this problem statement:

# 1) Exploratory Data Analysis:
Analyze and understand the data to identify patterns, relationships, and trends in the data by using Descriptive Statistics and Visualizations.
* Class imbalance visualization
* Distribution of `Amount` and `Time`
* Correlation matrix

# 2) Data Cleaning:
This might include standardization, handling the missing values and outliers in the data.
 * No missing values
 * Scaled `Amount` and `Time` using StandardScaler

# 3) Dealing with Imbalanced data:
This data set is highly imbalanced. The data should be balanced using the appropriate methods before moving onto model building.
* Used **SMOTE** (Synthetic Minority Oversampling Technique)
* Checked performance with and without resamplin

# 4) Feature Engineering:
Create new features or transform the existing features for better performance of the ML Models.

# 5) Model Selection:
Choose the most appropriate model that can be used for this project.
Tested multiple algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- SVM
- Naive Bayes
- Neural Network (MLPClassifier)

# 6) Model Training:
Split the data into train & test sets and use the train set to estimate the best model parameters.

# 7) Model Validation:
Evaluate the performance of the model on data that was not used during the training process. The goal is to estimate the model's ability to generalize to new, unseen data and to identify any issues with the model, such as overfitting.
Metrics used:
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC score
- PR-AUC score (important for imbalanced data)

# 8) Hyperparameter Tuning
Hyperparameter tuning is the process of optimizing the configuration parameters called hyperparameters that control the behavior of a machine learning algorithm — parameters that are not learned from the data but rather set before training begins.

# 9) Model Deployment:
Model deployment is the process of making a trained machine learning model available for use in a production environment.

# How to Run

# Install dependencies:
  pip install -r requirements.txt

# Run jupyter notebook
 run FindDefault (Prediction of Credit Card fraud).ipynb

# Run Main python file
python FindDefault (Prediction of Credit Card fraud).py

