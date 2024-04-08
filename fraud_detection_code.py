# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import vonmises
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load and preprocess the dataset
def load_dataset(transaction_data_path, label_data_path):
    # Load the datasets
    transaction_data = pd.read_csv(transaction_data_path)
    label_data = pd.read_csv(label_data_path)
    
    # Merge datasets on 'eventId'
    combined_data = pd.merge(transaction_data, label_data, on='eventId', how='left')
    
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
    threshold = 0.05

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
    
    return combined_data

# Split and balance the dataset
def split_and_balance_data(combined_data):
    # Separate the variables
    X = combined_data.drop('isFraud', axis=1)
    y = combined_data['isFraud']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Balance the training dataset using SMOTE
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test

# Train and evaluate the Random Forest model
def train_and_evaluate_random_forest(X_train_resampled, X_test, y_train_resampled, y_test):
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

    return accuracy, precision, recall, f1, roc_auc

# Train and evaluate the XGBoost model
def train_and_evaluate_xgboost(X_train_resampled, X_test, y_train_resampled, y_test):
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

    return accuracy, precision, recall, f1, roc_auc

# Execute the workflow
def main(transaction_data_path, label_data_path):
    # Load and preprocess the dataset
    combined_data = load_dataset(transaction_data_path, label_data_path)
    
    # Split and balance the dataset
    X_train_resampled, X_test, y_train_resampled, y_test = split_and_balance_data(combined_data)
    
    # Train and evaluate the Random Forest model
    rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = train_and_evaluate_random_forest(X_train_resampled, X_test, y_train_resampled, y_test)
    
    # Train and evaluate the XGBoost model
    xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_roc_auc = train_and_evaluate_xgboost(X_train_resampled, X_test, y_train_resampled, y_test)
    
    # Display the results
    print("Random Forest Model Evaluation Metrics:")
    print("Accuracy:", rf_accuracy)
    print("Precision:", rf_precision)
    print("Recall:", rf_recall)
    print("F1 Score:", rf_f1)
    print("ROC AUC Score:", rf_roc_auc)
    
    print("\nXGBoost Model Evaluation Metrics:")
    print("Accuracy:", xgb_accuracy)
    print("Precision:", xgb_precision)
    print("Recall:", xgb_recall)
    print("F1 Score:", xgb_f1)
    print("ROC AUC Score:", xgb_roc_auc)

# Execute the main function
if __name__ == "__main__":
    transaction_data_path = "transactions_obf.csv"
    label_data_path = "labels_obf.csv"
    main(transaction_data_path, label_data_path)
