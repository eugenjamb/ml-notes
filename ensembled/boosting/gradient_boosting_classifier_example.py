import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# ============================================================
# Gradient Boosting Classifier Example
# ============================================================
#
# What gradient boosting does:
# - builds learners sequentially like AdaBoost
# - each new learner tries to correct the residual mistakes of the current ensemble
# - uses gradient-based optimization ideas to decide the next step
#
# Why we use it:
# - it is one of the most important boosting families
# - it is often stronger and more flexible than basic AdaBoost


dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    # test_size=0.25 reserves 25% for final evaluation.
    test_size=0.25,
    # random_state=42 keeps the split reproducible.
    random_state=42,
    # stratify=y preserves class proportions.
    stratify=y,
)

# n_estimators=120 means 120 boosting stages.
# learning_rate=0.08 shrinks each stage's contribution.
# max_depth=2 controls the depth of the individual regression trees used internally.
# random_state=42 makes the stage sequence reproducible.
model = GradientBoostingClassifier(
    n_estimators=120,
    learning_rate=0.08,
    max_depth=2,
    random_state=42,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("GradientBoostingClassifier example")
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=dataset.target_names))

print(
    "Interpretation: gradient boosting improves the ensemble stage by stage,"
    " with each new tree focused on the remaining errors."
)
