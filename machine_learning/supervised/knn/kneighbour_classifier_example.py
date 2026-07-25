import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# KNN Classification Example
# ============================================================
#
# What this method does:
# - compares a new point to stored training examples
# - uses distance to find the nearest neighbors
# - predicts the class that appears most often nearby
#
# Why we use it:
# - it is intuitive for beginners
# - it works well when local similarity matters
# - it shows clearly why feature scaling matters


# Small learning dataset:
# Each row is [hours_studied, practice_tests_completed]
# Label 1 means "pass", label 0 means "fail".
X = np.array([
    [1, 0],
    [1, 1],
    [2, 1],
    [2, 2],
    [3, 1],
    [3, 2],
    [4, 2],
    [4, 3],
    [5, 3],
    [5, 4],
    [6, 4],
    [6, 5],
    [7, 4],
    [7, 5],
    [8, 5],
    [8, 6],
], dtype=float)

y = np.array([
    0, 0, 0, 0,
    0, 0, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
])


# Split the dataset into training data and test data.
# The model learns from the training data and we evaluate on test data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# Min-max scaling changes each feature to a 0-to-1 range.
# This is useful for KNN because KNN depends on distances.
# Without scaling, one feature could dominate just because its numbers are bigger.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Original training features:\n", X_train)
print("\nScaled training features:\n", X_train_scaled)


# Create the K-Nearest Neighbors classifier.
# n_neighbors=3 means the model looks at the 3 closest points.
# The final prediction is the majority vote among those neighbors.
model = KNeighborsClassifier(n_neighbors=3)


# Fit the model on the scaled training data.
model.fit(X_train_scaled, y_train)


# Predict labels for the test set.
y_pred = model.predict(X_test_scaled)


# Accuracy tells us how many predictions were correct.
# It is a simple metric for classification when false positives and
# false negatives are not weighted differently.
accuracy = accuracy_score(y_test, y_pred)
print("\nTest labels:      ", y_test)
print("Predicted labels: ", y_pred)
print("Accuracy:", round(accuracy, 3))


# Example new students to predict:
# [hours_studied, practice_tests_completed]
new_data = np.array([
    [2.5, 1.0],   # likely fail
    [4.5, 3.0],   # around the decision boundary
    [7.5, 5.5],   # likely pass
], dtype=float)


# Use the same scaler from training before making predictions.
# New data must be transformed in exactly the same way as training data.
new_data_scaled = scaler.transform(new_data)
new_predictions = model.predict(new_data_scaled)
new_probabilities = model.predict_proba(new_data_scaled)

print("\nNew data points:\n", new_data)
print("\nScaled new data points:\n", new_data_scaled)
print("\nPredictions for new data:", new_predictions)
print("Prediction probabilities:\n", new_probabilities)


# Extra explanation:
# Class 0 = fail
# Class 1 = pass
# predict_proba() shows how much of the local neighborhood belongs to each class.
