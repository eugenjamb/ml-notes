import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# Weighted KNN Example
# ============================================================
#
# What this method does:
# - compares uniform KNN against distance-weighted KNN
# - uniform weighting gives every neighbor the same vote
# - distance weighting gives closer neighbors more influence
#
# Why we use it:
# - it shows that not all neighbors need to matter equally
# - it helps near class boundaries where the closest points may be most informative
# - it is a simple hyperparameter choice inside the KNN family


# Small dataset created by hand:
# Each row is [minutes_exercised_per_day, sugary_drinks_per_week]
# Label 1 means "healthy", label 0 means "not healthy".
X = np.array([
    [10, 10],
    [12, 9],
    [14, 8],
    [16, 8],
    [18, 7],
    [20, 7],
    [30, 5],
    [35, 4],
    [40, 4],
    [45, 3],
    [50, 2],
    [55, 2],
    [60, 1],
    [65, 1],
    [70, 1],
    [75, 0],
], dtype=float)

y = np.array([
    0, 0, 0, 0,
    0, 0, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
])

# Keep the class ratio balanced in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=7,
    stratify=y,
)

# Scale features because KNN is distance-based.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare two versions of the same algorithm:
# - "uniform" treats each neighbor equally
# - "distance" emphasizes the nearest points
uniform_model = KNeighborsClassifier(n_neighbors=5, weights="uniform")
distance_model = KNeighborsClassifier(n_neighbors=5, weights="distance")

uniform_model.fit(X_train_scaled, y_train)
distance_model.fit(X_train_scaled, y_train)


uniform_predictions = uniform_model.predict(X_test_scaled)
distance_predictions = distance_model.predict(X_test_scaled)

# This is a useful comparison because both models see the same data
# and differ only in how they weight the neighborhood.
uniform_accuracy = accuracy_score(y_test, uniform_predictions)
distance_accuracy = accuracy_score(y_test, distance_predictions)

print("True labels:            ", y_test)
print("Uniform predictions:    ", uniform_predictions)
print("Distance predictions:   ", distance_predictions)
print("Uniform accuracy: ", round(uniform_accuracy, 3))
print("Distance accuracy:", round(distance_accuracy, 3))


# A point near the class boundary.
# Borderline points are where weighting choices matter most.
new_person = np.array([[28, 6]], dtype=float)
new_person_scaled = scaler.transform(new_person)

uniform_result = uniform_model.predict(new_person_scaled)[0]
distance_result = distance_model.predict(new_person_scaled)[0]

uniform_probability = uniform_model.predict_proba(new_person_scaled)[0]
distance_probability = distance_model.predict_proba(new_person_scaled)[0]

print("\nNew person:", new_person[0])
print("Uniform weighted class: ", uniform_result)
print("Distance weighted class:", distance_result)
print("Uniform probabilities: ", np.round(uniform_probability, 3))
print("Distance probabilities:", np.round(distance_probability, 3))
