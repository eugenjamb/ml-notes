from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# L1 or Lasso Regularization
# ============================================================
#
# L1 regularization adds a penalty based on the absolute value
# of the coefficients.
#
# Objective idea:
# loss + alpha * (|w1| + |w2| + ... + |wn|)
#
# Main effect:
# - discourages large coefficients
# - can drive some coefficients exactly to 0
# - this makes Lasso useful for feature selection
#
# Why we use it:
# - when we suspect some features are weak or noisy
# - when we want a simpler model that ignores less useful inputs
# - when we want an easier-to-interpret linear model


# make_regression() creates a synthetic dataset for linear regression.
# n_informative=4 means only 4 features contain real signal.
# The rest act more like distractors or weaker columns.
X, y, true_coefficients = make_regression(
    n_samples=220,
    n_features=12,
    n_informative=4,
    noise=18,
    coef=True,
    random_state=42,
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 reserves 25% for evaluation.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
)

# StandardScaler rescales each feature to a similar range.
# This matters because regularization penalizes coefficient size.
# If one feature has a much larger scale than the others, the penalty
# becomes harder to interpret fairly.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# LinearRegression is our baseline with no regularization penalty.
# It tries only to minimize prediction error on the training data.
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_predictions = linear_model.predict(X_test_scaled)


# Lasso adds the L1 penalty.
# alpha controls how strong that penalty is:
# - small alpha -> behaves more like ordinary linear regression
# - large alpha -> stronger shrinkage and more zero coefficients
# alpha=0.25 controls the strength of the L1 penalty.
# max_iter=10000 gives the optimizer enough steps to converge.
# random_state=42 makes stochastic parts reproducible if used internally.
lasso_model = Lasso(alpha=0.25, max_iter=10000, random_state=42)
lasso_model.fit(X_train_scaled, y_train)
lasso_predictions = lasso_model.predict(X_test_scaled)


comparison = pd.DataFrame(
    {
        "feature": feature_names,
        "true_signal": np.round(true_coefficients, 3),
        "linear_coef": np.round(linear_model.coef_, 3),
        "lasso_coef": np.round(lasso_model.coef_, 3),
    }
)

comparison["lasso_zeroed"] = comparison["lasso_coef"] == 0
comparison = comparison.sort_values("lasso_coef", key=np.abs, ascending=False)

print("L1 / Lasso regularization example")
print(
    "\nWhat Lasso does:"
    "\n- adds a penalty to coefficient size"
    "\n- shrinks weak features more aggressively than Ridge"
    "\n- can remove features by setting their coefficients to 0"
)

print(
    "\nWhy you would use it:"
    "\n- to reduce overfitting"
    "\n- to simplify the model"
    "\n- to keep only the strongest predictors"
)

print("\nCoefficient comparison:")
print(comparison.to_string(index=False))

print("\nModel comparison:")
print("Linear Regression R^2:", round(r2_score(y_test, linear_predictions), 3))
print("Lasso Regression R^2:", round(r2_score(y_test, lasso_predictions), 3))
print(
    "Linear Regression MSE:",
    round(mean_squared_error(y_test, linear_predictions), 2),
)
print("Lasso Regression MSE:", round(mean_squared_error(y_test, lasso_predictions), 2))
print(
    "Number of coefficients set to zero by Lasso:",
    int((lasso_model.coef_ == 0).sum()),
)

print(
    "\nInterpretation: L1 regularization shrinks weak features so aggressively"
    " that some become exactly 0. That is why Lasso is often described as a"
    " built-in feature-selection method."
)


plt.figure(figsize=(10, 5))
x_positions = np.arange(len(feature_names))
plt.bar(x_positions - 0.18, linear_model.coef_, width=0.36, label="Linear Regression")
plt.bar(x_positions + 0.18, lasso_model.coef_, width=0.36, label="Lasso")
plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
plt.xticks(x_positions, feature_names, rotation=45, ha="right")
plt.ylabel("Coefficient value")
plt.title("L1 Regularization Shrinks Some Coefficients To Zero")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "l1_lasso_coefficients.png", dpi=180)
plt.close()
