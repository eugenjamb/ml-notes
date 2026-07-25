import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# BaggingClassifier With K-Nearest Neighbors
# ============================================================
#
# What this example shows:
# - bagging can also wrap non-tree classifiers
# - here the base learner is KNeighborsClassifier
# - each bagged model is trained on a different bootstrap sample
#
# Why this matters:
# - bagging is a general ensemble idea, not just a tree trick
# - KNN is usually more stable than a decision tree
# - this makes it a good comparison point because gains may be smaller


dataset = load_wine()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 uses a quarter of the rows for testing.
    test_size=0.25,
    # random_state=42 keeps the split consistent across runs.
    random_state=42,
    # stratify=y keeps the class proportions similar in train and test.
    stratify=y,
)


# KNN needs scaling because it uses distances.
# StandardScaler() centers each feature and scales it to unit variance.
# n_neighbors=7 means each prediction uses the 7 closest training points.
single_knn = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7)),
    ]
)
single_knn.fit(X_train, y_train)
single_knn_predictions = single_knn.predict(X_test)


# We bag the whole pipeline so each bootstrap sample also fits its own scaler.
# estimator=Pipeline(...) means every bootstrap model includes scaling plus KNN.
# n_estimators=25 trains 25 bagged pipelines.
# bootstrap=True enables sampling with replacement.
# n_jobs=1 avoids parallel worker issues here.
bagging_knn = BaggingClassifier(
    estimator=Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7)),
        ]
    ),
    n_estimators=25,
    bootstrap=True,
    random_state=42,
    n_jobs=1,
)
bagging_knn.fit(X_train, y_train)
bagging_predictions = bagging_knn.predict(X_test)


single_accuracy = accuracy_score(y_test, single_knn_predictions)
bagging_accuracy = accuracy_score(y_test, bagging_predictions)

print("BaggingClassifier with KNeighborsClassifier")
print("\nSingle KNN accuracy:", round(single_accuracy, 3))
print("Bagged KNN accuracy:", round(bagging_accuracy, 3))

print("\nClassification report for bagged KNN:")
print(classification_report(y_test, bagging_predictions, target_names=dataset.target_names))

print(
    "Interpretation: bagging can still work with KNN, but the benefit is often"
    " smaller than with decision trees because KNN is usually less unstable."
)
