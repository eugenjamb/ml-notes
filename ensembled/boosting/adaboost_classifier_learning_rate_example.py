import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# AdaBoost Learning-Rate Comparison
# ============================================================
#
# This example focuses on one key AdaBoost hyperparameter: learning_rate.
#
# What learning_rate does:
# - it controls how strongly each weak learner contributes
# - smaller values make boosting more cautious
# - larger values make each stage more aggressive
#
# Why this matters:
# - boosting performance depends on both n_estimators and learning_rate
# - this is a clean way to show hyperparameter tradeoffs inside one algorithm


dataset = load_wine()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    # test_size=0.25 keeps 25% for evaluation.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
    # stratify=y preserves class proportions.
    stratify=y,
)

learning_rates = [0.1, 0.5, 1.0, 1.5]
results = []

for learning_rate in learning_rates:
    # max_depth=2 allows a slightly stronger weak learner than a stump.
    # n_estimators=60 fixes the number of boosting stages.
    # learning_rate is the parameter we are comparing across runs.
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
        n_estimators=60,
        learning_rate=learning_rate,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results.append(
        {
            "learning_rate": learning_rate,
            "test_accuracy": round(accuracy_score(y_test, predictions), 3),
        }
    )

results_table = pd.DataFrame(results)

print("AdaBoostClassifier learning-rate comparison")
print(results_table.to_string(index=False))
print(
    "\nInterpretation: learning_rate changes how aggressively AdaBoost updates"
    " the ensemble. It often works best when balanced with the number of estimators."
)
