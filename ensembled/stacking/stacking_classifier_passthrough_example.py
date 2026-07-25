import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Stacking Classifier With Passthrough
# ============================================================
#
# This example focuses on one important stacking parameter: passthrough.
#
# What passthrough does:
# - passthrough=False -> final estimator sees only base-model predictions
# - passthrough=True -> final estimator sees both the original features
#   and the base-model predictions
#
# Why this matters:
# - sometimes the meta-model benefits from access to the raw features too
# - this changes what information the second layer can use


dataset = load_wine()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 reserves 25% for testing.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
    # stratify=y preserves class balance.
    stratify=y,
)

estimators = [
    (
        "knn",
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    ),
    (
        "forest",
        RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42, n_jobs=1),
    ),
]

results = []

for passthrough_value in [False, True]:
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=5000, random_state=42),
        cv=5,
        passthrough=passthrough_value,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results.append(
        {
            "passthrough": passthrough_value,
            "test_accuracy": round(accuracy_score(y_test, predictions), 3),
        }
    )

results_table = pd.DataFrame(results)

print("StackingClassifier passthrough comparison")
print(results_table.to_string(index=False))
print(
    "\nInterpretation: passthrough changes whether the meta-model sees only"
    " first-layer predictions or also the original features."
)
