from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# Logistic Regression For Fraud Detection
# ============================================================
#
# What this method does:
# - models the probability that a transaction belongs to the fraud class
# - combines several numeric features into one decision boundary
# - outputs both class predictions and probabilities
#
# Why we use it:
# - it is a strong baseline for binary classification
# - coefficients are relatively easy to interpret
# - probability output is useful when fraud teams want risk scores


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "transactions_modified.csv"

# Load the data
transactions = pd.read_csv(DATA_PATH)

# Create isPayment field.
# This turns transaction type into a simple binary indicator the model can use.
transactions["isPayment"] = (
    (transactions["type"] == "PAYMENT")
    | (transactions["type"] == "DEBIT")
).astype(int)


# Create isMovement field.
# This groups transfer-like transaction types into another useful binary feature.
transactions["isMovement"] = (
    (transactions["type"] == "CASH_OUT")
    | (transactions["type"] == "TRANSFER")
).astype(int)


# Create accountDiff field.
# Feature engineering like this can expose patterns that are not obvious
# from the raw columns alone.
transactions["accountDiff"] = transactions["oldbalanceOrg"] - transactions["oldbalanceDest"]

# Create features and label variables.
# features = inputs used for prediction
# label = target we want to learn
features = transactions[["amount", "isPayment", "isMovement", "accountDiff"]]
print(features)
label = transactions["isFraud"]
print(label)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    features,
    label,
    # test_size=0.3 reserves 30% of the rows for evaluation.
    test_size=0.3,
    # random_state=5 makes the split reproducible.
    random_state=5,
)

# Normalize the feature variables, fitting only on the training data.
# This avoids leaking information from the test set into preprocessing.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)
print(X_test_scaled)

# Fit the model to the training data.
# LogisticRegression() predicts the probability of the fraud class.
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Score the model on the training data
training_score = model.score(X_train_scaled, y_train)
print(training_score)
# Score the model on the test data
test_score = model.score(X_test_scaled, y_test)
print(test_score)

# Print the model coefficients.
# In logistic regression, larger positive coefficients push predictions
# more strongly toward the fraud class after scaling.
print(model.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])
transaction4 = np.array([20000.40, 0.0, 1.0, 1000000.12])

# Combine new transactions into a single array
sample_transactions = np.stack([transaction1, transaction2, transaction3, transaction4])

# Normalize the new transactions
sample_transactions = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
sample_transactions_predicted = model.predict(sample_transactions)
print(sample_transactions_predicted)

# Show probabilities on the new transactions.
# Probability output is often more useful than a hard yes/no label.
sample_prob = model.predict_proba(sample_transactions)
print(sample_prob)
