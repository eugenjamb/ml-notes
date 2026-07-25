import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Logistic Regression Regularization Options
# ============================================================
#
# LogisticRegression in scikit-learn can use:
# - penalty=None -> no regularization
# - penalty="l1" -> L1 regularization
# - penalty="l2" -> L2 regularization
# - penalty="elasticnet" -> combination of L1 and L2
#
# What this method does:
# - predicts class probabilities for a binary target
# - turns those probabilities into class labels
# - uses penalties to control coefficient size and model complexity
#
# Why we use it:
# - it is a strong baseline for binary classification
# - regularization often improves generalization
# - different penalties give different tradeoffs between sparsity and stability


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
    # stratify=y preserves the class ratio in both splits.
    stratify=y,
)

# Scaling helps the optimizer and makes coefficient penalties more comparable.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Solver choice matters here:
# - lbfgs works for none and l2
# - liblinear works for l1 on binary tasks
# - saga supports elasticnet
models = [
    # penalty=None removes coefficient regularization.
    # max_iter=5000 gives the solver enough optimization steps.
    ("none", LogisticRegression(penalty=None, max_iter=5000, random_state=42)),
    # penalty="l2" applies ridge-style shrinkage.
    # C=1.0 is the inverse of regularization strength.
    # solver="lbfgs" supports this setting efficiently.
    ("l2", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, random_state=42)),
    # penalty="l1" can drive weak coefficients to zero.
    # solver="liblinear" supports L1 for binary tasks.
    ("l1", LogisticRegression(penalty="l1", C=1.0, solver="liblinear", max_iter=5000, random_state=42)),
    (
        "elasticnet",
        # penalty="elasticnet" mixes L1 and L2 regularization.
        # C=1.0 is the inverse regularization strength.
        # solver="saga" is needed for elastic net support.
        # l1_ratio=0.5 mixes L1 and L2 evenly.
        LogisticRegression(
            penalty="elasticnet",
            C=1.0,
            solver="saga",
            l1_ratio=0.5,
            max_iter=5000,
            random_state=42,
        ),
    ),
]

results = []

for model_name, model in models:
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    results.append(
        {
            "model": model_name,
            "accuracy": round(accuracy_score(y_test, predictions), 3),
            "f1": round(f1_score(y_test, predictions), 3),
            "nonzero_coefficients": int((~pd.Series(model.coef_[0]).round(10).eq(0)).sum()),
            "coefficient_l1_norm": round(float(abs(model.coef_[0]).sum()), 3),
        }
    )

results_table = pd.DataFrame(results).sort_values("accuracy", ascending=False)

print("LogisticRegression regularization options")
print(
    "\nWhat the penalties do:"
    "\n- none: no coefficient penalty"
    "\n- l2: shrinks coefficients smoothly"
    "\n- l1: can push weak coefficients to zero"
    "\n- elasticnet: mixes l1 and l2 behavior"
)

print(
    "\nWhy this comparison matters:"
    "\n- the same algorithm family can behave differently depending on penalty"
    "\n- some penalties favor simpler models"
    "\n- others favor coefficient stability over sparsity"
)

print("\nModel comparison:")
print(results_table.to_string(index=False))

print(
    "\nInterpretation: LogisticRegression is not just one model setting."
    " The penalty choice affects both generalization and coefficient shape."
)
