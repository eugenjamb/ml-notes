import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# Stacking Classifier Example
# ============================================================
#
# What stacking does:
# - trains several different base models
# - uses their predictions as inputs to a final meta-model
# - lets the meta-model learn how to combine the base learners
#
# Why we use it:
# - different algorithms can capture different patterns
# - stacking can outperform any single base learner
# - it is more flexible than plain voting because the final model learns
#   how to weight the earlier models


dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 reserves 25% for evaluation.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
    # stratify=y preserves class balance.
    stratify=y,
)


# These are the base learners.
# Each one sees the original training features.
estimators = [
    (
        "knn",
        Pipeline(
            [
                # StandardScaler() is needed because KNN uses distances.
                ("scaler", StandardScaler()),
                # n_neighbors=7 means KNN uses the 7 nearest examples.
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
    ),
    (
        "svc",
        Pipeline(
            [
                # StandardScaler() is also useful for SVMs.
                ("scaler", StandardScaler()),
                # kernel="rbf" gives a non-linear SVM boundary.
                # C=1.0 is the default regularization strength.
                ("model", SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
            ]
        ),
    ),
    (
        "forest",
        # n_estimators=80 means the forest averages 80 trees.
        # max_depth=5 limits tree complexity.
        RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42, n_jobs=1),
    ),
]


# The final estimator is the meta-model.
# It learns from the base models' predictions.
stacking_model = StackingClassifier(
    estimators=estimators,
    # final_estimator=... is the meta-model that combines base predictions.
    # LogisticRegression is a common simple choice for this layer.
    final_estimator=LogisticRegression(max_iter=5000, random_state=42),
    # cv=5 means the stacking layer is built from 5-fold cross-validated predictions.
    cv=5,
    # passthrough=False means the final estimator sees only base-model predictions.
    passthrough=False,
    n_jobs=1,
)
stacking_model.fit(X_train, y_train)
predictions = stacking_model.predict(X_test)

print("StackingClassifier example")
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=dataset.target_names))

print(
    "Interpretation: stacking lets a final model learn how to combine KNN,"
    " SVM, and random forest predictions instead of just averaging or voting."
)
