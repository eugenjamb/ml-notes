import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ============================================================
# Gradient Boosting Regressor Example
# ============================================================
#
# This is one of the most common regression ensemble models.
#
# What it does:
# - builds shallow trees one after another
# - each new tree predicts the residual error left by the current ensemble
# - the ensemble adds those small corrections together
#
# Why we use it:
# - it can model non-linear relationships well
# - it often performs strongly on tabular data


dataset = load_diabetes()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    # test_size=0.25 reserves 25% of the rows for testing.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
)

# n_estimators=180 means 180 sequential boosting stages.
# learning_rate=0.05 makes each tree's contribution smaller and more gradual.
# max_depth=2 keeps each tree shallow so the ensemble learns in small steps.
model = GradientBoostingRegressor(
    n_estimators=180,
    learning_rate=0.05,
    max_depth=2,
    random_state=42,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("GradientBoostingRegressor example")
print("Test R^2:", round(r2_score(y_test, predictions), 3))
print("Test MSE:", round(mean_squared_error(y_test, predictions), 2))
print(
    "Interpretation: gradient boosting regression works by repeatedly fitting"
    " new trees to the remaining numeric errors."
)
