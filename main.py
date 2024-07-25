import pandas as pd
import numpy as np
from faker import Faker
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

fake = Faker()

# Generate synthetic data
num_records = 1000

transaction_types = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
statuses = ["Completed", "Pending", "Failed"]
devices = ["Desktop", "Mobile", "Tablet"]
browsers = ["Chrome", "Firefox", "Safari", "Edge"]
geolocations = ["US", "UK", "CA", "DE", "FR"]

data = []
for _ in range(num_records):
    transaction_type = random.choice(transaction_types)
    amount = round(random.uniform(1.0, 10000.0), 2)
    oldBal = round(random.uniform(0.0, 20000.0), 2)
    newBal = oldBal + amount if transaction_type == "CASH_IN" else oldBal - amount
    oldBalDest = round(random.uniform(0.0, 20000.0), 2)
    newBalDest = oldBalDest - amount if transaction_type == "CASH_OUT" else oldBalDest + amount

    # Define fraud and flagged fraud status based on balance changes
    isFraud = 1 if abs((newBal - oldBal)==(newBalDest - oldBalDest)) else 0

    timestamp = fake.date_time_this_year()
    device = random.choice(devices)
    browser = random.choice(browsers)
    geolocation = random.choice(geolocations)
    
    data.append([
        transaction_type, amount, oldBal, newBal, oldBalDest, newBalDest,
        isFraud, timestamp, device, browser, geolocation
    ])

columns = [
    "Type", "Amount", "oldBal", "newBal", "oldBalDest", "newBalDest", "isFraud",
    "Timestamp", "DeviceType", "Browser", "Geolocation"
]
df = pd.DataFrame(data, columns=columns)

# Save synthetic data to CSV
df.to_csv("synthetic_fraud_detection_data.csv", index=False)

# Load the data
file_path = 'synthetic_fraud_detection_data.csv'
data = pd.read_csv(file_path)

# Feature engineering
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Hour'] = data['Timestamp'].dt.hour
data['Day'] = data['Timestamp'].dt.day
data['Month'] = data['Timestamp'].dt.month
data['IsBusinessHours'] = data['Hour'].apply(lambda x: 1 if 9 <= x <= 17 else 0)

data['NormalizedAmount'] = data['Amount'] / (data['oldBal'] + 1)  # Adding 1 to avoid division by zero
data['AmountPercentage'] = (data['Amount'] / (data['oldBal'] + 1)) * 100

data['OriginBalanceChange'] = data['newBal'] - data['oldBal']
data['DestBalanceChange'] = data['newBalDest'] - data['oldBalDest']

data['BalanceDifference'] = np.abs(data['OriginBalanceChange'] - data['DestBalanceChange'])

top_devices = data['DeviceType'].value_counts().index[:3]
data['IsCommonDevice'] = data['DeviceType'].apply(lambda x: 1 if x in top_devices else 0)

top_browsers = data['Browser'].value_counts().index[:3]
data['IsCommonBrowser'] = data['Browser'].apply(lambda x: 1 if x in top_browsers else 0)

# Drop unnecessary columns
data = data.drop(columns=['Timestamp'])

data['Type'] = data['Type'].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
data['DeviceType'] = data['DeviceType'].astype('category').cat.codes
data['Browser'] = data['Browser'].astype('category').cat.codes
data['Geolocation'] = data['Geolocation'].astype('category').cat.codes

data = data.dropna()

# Save engineered data to CSV
data.to_csv('engineered_fraud_detection_data.csv', index=False)

# Split data into features and labels
X = data.drop(columns=['isFraud'])
y = data['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance the training set using SMote
smote = SMOTE(random_state=42, k_neighbors=2)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Save the train and test sets to CSV
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False, header=True)
y_test.to_csv('y_test.csv', index=False, header=True)

# Train a RandomForest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_balanced, y_train_balanced)

# Predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_balanced, y_train_balanced)
best_rf_classifier = grid_search.best_estimator_

# Final predictions and evaluation
y_pred = best_rf_classifier.predict(X_test)
y_pred_proba = best_rf_classifier.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print(f'AUC-ROC: {roc_auc:.2f}')

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Save the model
joblib.dump(best_rf_classifier, 'best_rf_classifier.pkl')

# Feature importances
feature_importances = best_rf_classifier.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Load the saved model
model = joblib.load('best_rf_classifier.pkl')

# Retrieve feature names from training data
feature_names = X.columns

# Define a fraudulent test case
test_case = {
    "Type": 1,  # CASH_OUT
    "Amount": 5000.0,
    "oldBal": 10000.0,
    "newBal": 10000.0,
    "oldBalDest": 15000.0,
    "newBalDest": 10000.0,
    "DeviceType": 1,  # Mobile
    "Browser": 0,  # Chrome
    "Geolocation": 0,  # US
    "Hour": 14,
    "Day": 15,
    "Month": 7,
    "IsBusinessHours": 1,
    "NormalizedAmount": 5000.0 / (10000.0 + 1),
    "AmountPercentage": (5000.0 / (10000.0 + 1)) * 100,
    "IsCommonDevice": 1,
    "IsCommonBrowser": 1
}

# Compute additional features required for prediction
test_case["OriginBalanceChange"] = test_case["newBal"] - test_case["oldBal"]
test_case["DestBalanceChange"] = test_case["newBalDest"] - test_case["oldBalDest"]
test_case["BalanceDifference"] = abs(test_case["OriginBalanceChange"] - test_case["DestBalanceChange"])

# Convert test case to DataFrame using the correct feature order
test_case_df = pd.DataFrame([test_case], columns=feature_names)

# Predict and print result
predicted_fraud = model.predict(test_case_df)
print(f'The test case is predicted to be {"fraudulent" if predicted_fraud[0] == 1 else "not fraudulent"}.')