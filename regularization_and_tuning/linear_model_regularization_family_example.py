from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Linear Regression Regularization Family
# ============================================================
#
# This file compares four related linear regression models:
# - LinearRegression -> no regularization penalty
# - Ridge -> L2 regularization
# - Lasso -> L1 regularization
# - ElasticNet -> blend of L1 and L2 regularization
#
# What these methods do:
# - all four models try to predict a numeric target
# - the regularized versions add penalties to discourage overly large coefficients
# - different penalties change how the model balances simplicity and flexibility
#
# Why we use them:
# - to reduce overfitting
# - to stabilize coefficients when features overlap
# - to simplify models when some predictors are weak or noisy


# Create a regression problem with several noisy features.
# Only a subset of the columns contains strong signal.
X, y, true_coefficients = make_regression(
    n_samples=260,
    n_features=14,
    n_informative=5,
    noise=20,
    coef=True,
    random_state=42,
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 reserves 25% for testing.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
)

# Scale features so regularization treats them fairly.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Build the model family.
# ElasticNet uses l1_ratio to decide how much L1 vs L2 penalty it keeps.
models = {
    # LinearRegression() has no regularization penalty.
    "linear": LinearRegression(),
    # alpha=5.0 controls the L2 penalty strength for Ridge.
    "ridge_l2": Ridge(alpha=5.0),
    # alpha=0.2 controls the L1 penalty strength for Lasso.
    # max_iter=10000 gives the optimizer enough iterations.
    "lasso_l1": Lasso(alpha=0.2, max_iter=10000, random_state=42),
    # alpha=0.2 sets the overall regularization strength.
    # l1_ratio=0.5 splits the penalty evenly between L1 and L2.
    # max_iter=10000 gives the optimizer enough time to converge.
    "elastic_net": ElasticNet(alpha=0.2, l1_ratio=0.5, max_iter=10000, random_state=42),
}

results = []
coefficient_table = pd.DataFrame({"feature": feature_names, "true_signal": np.round(true_coefficients, 3)})

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    results.append(
        {
            "model": model_name,
            "test_r2": round(r2_score(y_test, predictions), 3),
            "test_mse": round(mean_squared_error(y_test, predictions), 2),
            "coef_l1_norm": round(float(np.abs(model.coef_).sum()), 3),
            "coef_l2_norm": round(float(np.linalg.norm(model.coef_)), 3),
            "zero_coefficients": int((np.isclose(model.coef_, 0.0)).sum()),
        }
    )

    coefficient_table[model_name] = np.round(model.coef_, 3)

results_table = pd.DataFrame(results).sort_values("test_r2", ascending=False)

print("Linear model regularization family")
print(
    "\nWhat each model is for:"
    "\n- LinearRegression: baseline with no penalty"
    "\n- Ridge: shrinks coefficients smoothly with L2"
    "\n- Lasso: can remove weak features with L1"
    "\n- ElasticNet: mixes both behaviors"
)

print(
    "\nWhy ElasticNet matters:"
    "\n- sometimes Ridge keeps too many weak features"
    "\n- sometimes Lasso is too aggressive"
    "\n- ElasticNet gives a compromise between stability and sparsity"
)

print("\nModel comparison:")
print(results_table.to_string(index=False))

print("\nCoefficient comparison:")
print(coefficient_table.to_string(index=False))

print(
    "\nInterpretation: the regularized models usually trade a small amount of fit"
    " for better coefficient control. Ridge mainly shrinks, Lasso can zero out,"
    " and ElasticNet sits in between."
)


plot_order = ["linear", "ridge_l2", "lasso_l1", "elastic_net"]
x_positions = np.arange(len(feature_names))
bar_width = 0.2

plt.figure(figsize=(12, 6))
for offset, model_name in enumerate(plot_order):
    plt.bar(
        x_positions + (offset - 1.5) * bar_width,
        coefficient_table[model_name],
        width=bar_width,
        label=model_name,
    )

plt.axhline(0, color="black", linewidth=0.8, alpha=0.4)
plt.xticks(x_positions, feature_names, rotation=45, ha="right")
plt.ylabel("Coefficient value")
plt.title("Linear Model Coefficients Under Different Regularizers")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "linear_model_regularization_family.png", dpi=180)
plt.close()
