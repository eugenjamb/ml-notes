import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# ============================================================
# AdaBoost Regressor Example
# ============================================================
#
# Boosting also works for regression.
# Instead of voting on classes, the ensemble combines numeric predictions.
#
# Why this matters:
# - it shows that boosting is a general ensemble idea
# - the same sequential correction idea can be used for continuous targets


dataset = load_diabetes()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    # test_size=0.25 keeps 25% of the rows for regression testing.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
)

# max_depth=4 controls how complex each weak regression tree can be.
# n_estimators=120 means 120 boosting stages.
# learning_rate=0.5 scales each stage's contribution.
model = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=4, random_state=42),
    n_estimators=120,
    learning_rate=0.5,
    random_state=42,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("AdaBoostRegressor example")
print("Test R^2:", round(r2_score(y_test, predictions), 3))
print("Test MSE:", round(mean_squared_error(y_test, predictions), 2))
print(
    "Interpretation: AdaBoostRegressor builds a sequence of weak regression"
    " trees and combines them into a stronger numeric predictor."
)
