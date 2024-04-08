# Import relevant libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the datasets
transaction_data_path = "transactions_obf.csv"
label_data_path = "labels_obf.csv"
transaction_data = pd.read_csv(transaction_data_path)
label_data = pd.read_csv(label_data_path)

# Merge datasets on 'eventId'
combined_data = pd.merge(transaction_data, label_data, on='eventId', how='left')

# -------------Feature Engineering-------------

# Extract useful features from datetime column
combined_data['transactionTime'] = pd.to_datetime(combined_data['transactionTime'])
combined_data.sort_values(by='transactionTime', inplace=True)
combined_data['transactionHour'] = combined_data['transactionTime'].dt.hour # Hour of the day
combined_data['transactionDayOfWeek'] = combined_data['transactionTime'].dt.dayofweek # Day of the week
combined_data['transactionDayOfMonth'] = combined_data['transactionTime'].dt.day # transaction day
combined_data['transactionMonth'] = combined_data['transactionTime'].dt.month # transaction month
combined_data['transactionYear'] = combined_data['transactionTime'].dt.year # transaction year

# Transaction Recency (How long ago a transaction took place?)
combined_data['recency'] = (combined_data['transactionTime'].max() - combined_data['transactionTime']).dt.total_seconds()

# Transaction Frequency (number of transactions per unit of time)
combined_data['frequency'] = combined_data.groupby('accountNumber')['accountNumber'].transform('count')

# Time since Last Transaction (In seconds)
combined_data['timeSinceLastTransaction'] = combined_data.groupby('accountNumber')['transactionTime'].diff().dt.total_seconds().fillna(0)

# Merchant Frequency
combined_data['merchantFrequency'] = combined_data.groupby('merchantId')['merchantId'].transform('count')

# Account Age (Since account creation time is not available, the time of the first transaction will be utilized)
combined_data['accountAge'] = (combined_data['transactionTime'] - combined_data.groupby('accountNumber')['transactionTime'].transform('min')).dt.total_seconds()

# Transaction History
combined_data['totalTransactionCount'] = combined_data.groupby('accountNumber')['accountNumber'].transform('count')
combined_data['totalTransactionAmount'] = combined_data.groupby('accountNumber')['transactionAmount'].transform('sum')
combined_data['avgTransactionAmount'] = combined_data.groupby('accountNumber')['transactionAmount'].transform('mean')

# Transaction Amount Deviation
combined_data['transactionAmountDeviation'] = combined_data['transactionAmount'] - combined_data.groupby('accountNumber')['transactionAmount'].transform('mean')

# Calculate mean and standard deviation of transaction times for each account holder
mean_transaction_time = combined_data.groupby('accountNumber')['transactionTime'].apply(lambda x: x.dt.hour.mean())
std_transaction_time = combined_data.groupby('accountNumber')['transactionTime'].apply(lambda x: x.dt.hour.std())

# Fit Von Mises Distribution to transaction times of each account holder
for account_number, (mean, std) in zip(mean_transaction_time.index, zip(mean_transaction_time, std_transaction_time)):
    data = combined_data.loc[combined_data['accountNumber'] == account_number, 'transactionTime'].dt.hour
    model = vonmises.fit(data, fscale=std)
    combined_data.loc[combined_data['accountNumber'] == account_number, 'vonmises_pdf'] = vonmises.pdf(data, *model)

# Threshold for flagging transactions
threshold = 0.05  # Adjust as needed

# Flag transactions outside usual transaction times
combined_data['unusual_transaction_time'] = combined_data['vonmises_pdf'] < threshold

# Derive 'isFraud' feature based on the presence of fraud flags
combined_data['isFraud'] = combined_data['reportedTime'].notnull().astype(int)

# Replace missing values in 'merchantZip' column with 'UNKNOWN'
combined_data['merchantZip'].fillna('UNKNOWN', inplace=True)

# Drop unnecessary columns
combined_data.drop(['transactionTime', 'eventId', 'reportedTime', 'accountNumber', 'merchantId', 'merchantZip'], axis=1, inplace=True)

# Convert boolean variable into numerical representation
combined_data['unusual_transaction_time'] = combined_data['unusual_transaction_time'].astype(int)

# ------------------Data Splitting----------------------

# Separate the variables
X = combined_data.drop('isFraud', axis=1)
y = combined_data['isFraud']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------Data Balancing---------------------

# Balance the training dataset using SMOTE
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# --------------------Random Forest Model Training and Predictions-----------------------

# Initialize and train the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Predictions on the test set
y_pred = rf_model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Display the result
print("Random Forest Model Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

# ---------------XGBoost Model Training and Predictions-------------------

# Create an instance of XGBoost model
xgb_model = xgb.XGBClassifier()

# Train the model on your training data
xgb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on your test data
y_pred = xgb_model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Display the results
print("\nXGBoost Model Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

### -------------------------Hyperparameter Tuning---------------------------------

# Define hyperparameter grids for each algorithm
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Create GridSearchCV instances for each algorithm
rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5)
xgb_grid_search = GridSearchCV(xgb.XGBClassifier(), xgb_param_grid, cv=5)

# Perform Grid Search and get the best hyperparameters
rf_grid_search.fit(X_train_resampled, y_train_resampled)
rf_best_params = rf_grid_search.best_params_
rf_best_score = rf_grid_search.best_score_

xgb_grid_search.fit(X_train_resampled, y_train_resampled)
xgb_best_params = xgb_grid_search.best_params_
xgb_best_score = xgb_grid_search.best_score_

# Print the best hyperparameters and their corresponding scores
print("Random Forest Best Parameters:", rf_best_params)
print("Random Forest Best Score:", rf_best_score)

print("XGBoost Best Parameters:", xgb_best_params)
print("XGBoost Best Score:", xgb_best_score)

# Use the best hyperparameters for model fitting
best_rf_model = RandomForestClassifier(**rf_best_params, random_state = 0)
best_xgb_model = xgb.XGBClassifier(**xgb_best_params, random_state=0)

best_rf_model.fit(X_train_resampled, y_train_resampled)
best_xgb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions using the best models
rf_predictions = best_rf_model.predict(X_test)
xgb_predictions = best_xgb_model.predict(X_test)

# Calculate and display accuracy for each algorithm
rf_accuracy = accuracy_score(y_test, rf_predictions)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
rf_precision = precision_score(y_test, rf_predictions)
xgb_precision = precision_score(y_test, xgb_predictions)
rf_recall = recall_score(y_test, rf_predictions)
xgb_recall = recall_score(y_test, xgb_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
xgb_f1 = f1_score(y_test, xgb_predictions)
rf_roc_auc = roc_auc_score(y_test, rf_predictions)
xgb_roc_auc = roc_auc_score(y_test, xgb_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print("XGBoost Accuracy:", xgb_accuracy)
print("Random Forest Precision:", rf_precision)
print("XGBoost Precision:", xgb_precision)
print("Random Forest Recall:", rf_recall)
print("XGBoost Recall:", xgb_recall)
print("Random Forest F1-Score:", rf_f1)
print("XGBoost F1-Score:", xgb_f1)
print("Random Forest Roc_Auc:", rf_roc_auc)
print("XGBoost Roc_Auc:", xgb_roc_auc)
