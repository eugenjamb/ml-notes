from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# L2 or Ridge Regularization
# ============================================================
#
# L2 regularization adds a penalty based on squared coefficients.
#
# Objective idea:
# loss + alpha * (w1^2 + w2^2 + ... + wn^2)
#
# Main effect:
# - discourages very large coefficients
# - usually keeps all features in the model
# - helps when features are noisy or strongly correlated
#
# Why we use it:
# - when we want to reduce overfitting without dropping features
# - when many predictors contain some useful signal
# - when features overlap heavily and coefficients become unstable


# effective_rank makes the feature matrix more correlated.
# That is useful here because Ridge often helps when predictors
# are related to one another.
X, y = make_regression(
    n_samples=260,
    n_features=10,
    n_informative=5,
    noise=24,
    effective_rank=4,
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

# Scale the features so the penalty affects them on comparable footing.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Baseline with no regularization.
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_predictions = linear_model.predict(X_test_scaled)

# Test several alpha values to see how stronger regularization
# changes both performance and coefficient size.
ridge_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_results = []

for alpha in ridge_alphas:
    # alpha controls how strong the L2 penalty is.
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_scaled, y_train)
    predictions = ridge_model.predict(X_test_scaled)
    ridge_results.append(
        {
            "alpha": alpha,
            "test_r2": round(r2_score(y_test, predictions), 3),
            "test_mse": round(mean_squared_error(y_test, predictions), 2),
            "coef_l2_norm": round(float(np.linalg.norm(ridge_model.coef_)), 3),
            "zero_coefficients": int((ridge_model.coef_ == 0).sum()),
        }
    )

ridge_table = pd.DataFrame(ridge_results)

# alpha=10.0 is the chosen setting for the coefficient comparison section.
chosen_ridge = Ridge(alpha=10.0)
chosen_ridge.fit(X_train_scaled, y_train)

coefficient_table = pd.DataFrame(
    {
        "feature": feature_names,
        "linear_coef": np.round(linear_model.coef_, 3),
        "ridge_coef_alpha_10": np.round(chosen_ridge.coef_, 3),
    }
).sort_values("ridge_coef_alpha_10", key=np.abs, ascending=False)

print("L2 / Ridge regularization example")
print(
    "\nWhat Ridge does:"
    "\n- adds a penalty based on squared coefficient size"
    "\n- shrinks coefficients smoothly"
    "\n- usually keeps every feature in the model"
)

print(
    "\nWhy you would use it:"
    "\n- to control overfitting"
    "\n- to stabilize coefficients when features overlap"
    "\n- to keep all predictors while making the model less sensitive"
)

print("\nBaseline Linear Regression:")
print("R^2:", round(r2_score(y_test, linear_predictions), 3))
print("MSE:", round(mean_squared_error(y_test, linear_predictions), 2))

print("\nRidge results across alpha values:")
print(ridge_table.to_string(index=False))

print("\nCoefficient comparison (Linear vs Ridge alpha=10):")
print(coefficient_table.to_string(index=False))

print(
    "\nInterpretation: Ridge usually makes coefficients smaller without forcing"
    " them to exactly zero. That helps control variance while still keeping all"
    " predictors in the model."
)


plt.figure(figsize=(10, 5))
plt.plot(ridge_table["alpha"], ridge_table["coef_l2_norm"], marker="o", linewidth=2)
plt.xscale("log")
plt.xlabel("alpha (log scale)")
plt.ylabel("L2 norm of coefficients")
plt.title("Ridge Regularization Shrinks Coefficients As Alpha Grows")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "l2_ridge_coefficient_norms.png", dpi=180)
plt.close()
