import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# AdaBoost Classifier With Decision Stumps
# ============================================================
#
# What boosting does:
# - trains weak learners one after another
# - pays more attention to mistakes from earlier learners
# - combines many weak learners into one stronger model
#
# Why AdaBoost is useful:
# - it is one of the classic boosting methods
# - it often starts with very small trees called decision stumps
# - it shows how sequential reweighting differs from bagging


dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    # test_size=0.25 reserves 25% of the data for testing.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
    # stratify=y keeps the class balance similar in each split.
    stratify=y,
)


# A decision stump is just a depth-1 tree.
# By itself it is weak, but AdaBoost can combine many of them.
# max_depth=1 forces the tree to be a one-split weak learner.
stump = DecisionTreeClassifier(max_depth=1, random_state=42)

# estimator=stump chooses the weak learner family.
# n_estimators=80 means 80 weak learners are added in sequence.
# learning_rate=0.8 scales how strongly each learner contributes.
# random_state=42 makes the boosting sequence reproducible.
model = AdaBoostClassifier(
    estimator=stump,
    n_estimators=80,
    learning_rate=0.8,
    random_state=42,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("AdaBoostClassifier with decision stumps")
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=dataset.target_names))

print(
    "Interpretation: AdaBoost turns many weak stumps into a stronger classifier"
    " by focusing later learners on examples that earlier stumps found difficult."
)
