import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# KNN Regression Example
# ============================================================
#
# What this method does:
# - finds the nearest training points to a new example
# - averages their target values instead of voting on a class
# - predicts a continuous number, not a category
#
# Why we use it:
# - it is a natural extension of KNN classification
# - it works when nearby examples should have similar numeric outcomes
# - it gives an intuitive introduction to non-parametric regression


# Small dataset created by hand:
# Each row is [hours_studied, practice_tests_completed]
# Target value is the exam score from 0 to 100.
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
    42, 45, 48, 52,
    55, 58, 63, 67,
    71, 74, 79, 83,
    85, 88, 92, 95,
], dtype=float)

# Split data so we can evaluate prediction error on unseen examples.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)


# Scaling matters for KNN because it uses distance.
# Features should be on comparable scales before neighbor search.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# n_neighbors=3 means the model predicts each score from the
# average of the 3 closest training examples.
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train_scaled, y_train)


# Mean absolute error tells us the average size of the mistake
# in the original target units, which is easy to interpret.
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)

print("Test scores:     ", y_test)
print("Predicted scores:", np.round(y_pred, 1))
print("Mean absolute error:", round(mae, 2))


# Predict scores for new students.
# This lets us use the fitted neighborhood structure on fresh examples.
new_students = np.array([
    [2.5, 1.5],
    [4.5, 3.0],
    [7.5, 5.5],
], dtype=float)

new_students_scaled = scaler.transform(new_students)
predicted_scores = model.predict(new_students_scaled)

print("\nNew students:\n", new_students)
print("Predicted exam scores:", np.round(predicted_scores, 1))
