import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# BaggingClassifier With Decision Trees
# ============================================================
#
# What this example shows:
# - a single decision tree can be quite unstable
# - bagging trains many trees on different bootstrap samples
# - the final classifier combines their votes
#
# Why this matters:
# - decision trees often have high variance
# - bagging is especially helpful for high-variance base learners
# - this is one of the clearest examples of why ensemble methods work


dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 keeps 25% of the data for evaluation.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
    # stratify=y preserves the class balance in both splits.
    stratify=y,
)


# First train one tree so we have a baseline.
# max_depth=5 limits how many split levels the tree can grow.
# random_state=42 makes the tree-building process reproducible.
single_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
single_tree.fit(X_train, y_train)
single_tree_predictions = single_tree.predict(X_test)


# BaggingClassifier builds many trees on different bootstrap samples.
# bootstrap=True means each estimator sees a sample drawn with replacement.
# estimator=... picks the base learner that will be repeated.
# n_estimators=50 means the ensemble will train 50 separate trees.
# random_state=42 fixes the bootstrap sampling sequence.
# n_jobs=1 keeps execution single-process in this environment.
bagging_tree = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
    n_estimators=50,
    bootstrap=True,
    random_state=42,
    n_jobs=1,
)
bagging_tree.fit(X_train, y_train)
bagging_predictions = bagging_tree.predict(X_test)


single_accuracy = accuracy_score(y_test, single_tree_predictions)
bagging_accuracy = accuracy_score(y_test, bagging_predictions)

print("BaggingClassifier with DecisionTreeClassifier")
print("\nSingle tree accuracy:", round(single_accuracy, 3))
print("Bagged trees accuracy:", round(bagging_accuracy, 3))

print("\nClassification report for bagged trees:")
print(classification_report(y_test, bagging_predictions, target_names=dataset.target_names))

print(
    "Interpretation: bagging usually helps decision trees because each tree is"
    " sensitive to changes in the training data. Voting across many bootstrap"
    " trees reduces that instability."
)
