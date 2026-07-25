import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# BaggingClassifier With Logistic Regression
# ============================================================
#
# What this example shows:
# - bagging can use LogisticRegression as the base classifier
# - each model trains on a different bootstrap sample
# - the final ensemble uses voting across those logistic models
#
# Why this matters:
# - logistic regression is usually more stable than a tree
# - stable learners often gain less from bagging
# - this helps explain that ensemble performance depends on the base learner


dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 reserves 25% of the data for evaluation.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
    # stratify=y keeps class proportions stable across the split.
    stratify=y,
)


# Logistic regression benefits from scaling.
# StandardScaler() puts features on a comparable numeric scale.
# max_iter=5000 gives the optimizer enough steps to converge.
# random_state=42 makes stochastic pieces reproducible if used by the solver.
single_logistic = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42)),
    ]
)
single_logistic.fit(X_train, y_train)
single_logistic_predictions = single_logistic.predict(X_test)


# Bag the whole pipeline so each bootstrap model handles preprocessing correctly.
# n_estimators=25 means 25 logistic models are trained on different bootstrap samples.
# bootstrap=True turns the ensemble into bagging instead of plain repeated fitting.
bagging_logistic = BaggingClassifier(
    estimator=Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=42)),
        ]
    ),
    n_estimators=25,
    bootstrap=True,
    random_state=42,
    n_jobs=1,
)
bagging_logistic.fit(X_train, y_train)
bagging_predictions = bagging_logistic.predict(X_test)


single_accuracy = accuracy_score(y_test, single_logistic_predictions)
bagging_accuracy = accuracy_score(y_test, bagging_predictions)

print("BaggingClassifier with LogisticRegression")
print("\nSingle logistic regression accuracy:", round(single_accuracy, 3))
print("Bagged logistic regression accuracy:", round(bagging_accuracy, 3))

print("\nClassification report for bagged logistic regression:")
print(classification_report(y_test, bagging_predictions, target_names=dataset.target_names))

print(
    "Interpretation: bagging logistic regression can work, but the improvement"
    " is often limited because logistic regression is already a relatively"
    " stable learner compared with a decision tree."
)
