import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Stacking Regressor Example
# ============================================================
#
# Stacking also works for regression.
# Instead of combining class predictions, the meta-model learns how to combine
# numeric predictions from several base regressors.
#
# Why we use it:
# - different regressors can capture different aspects of the data
# - the final regressor can learn when to trust which base model


dataset = load_diabetes()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 reserves 25% for evaluation.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
)

estimators = [
    (
        "knn",
        Pipeline(
            [
                # StandardScaler() is helpful because KNN regression uses distances.
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=7)),
            ]
        ),
    ),
    (
        "ridge",
        # alpha=1.0 controls the strength of L2 regularization.
        Ridge(alpha=1.0),
    ),
    (
        "forest",
        # n_estimators=100 means the regressor averages 100 trees.
        # max_depth=6 limits the complexity of each tree.
        RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=1),
    ),
]

stacking_model = StackingRegressor(
    estimators=estimators,
    # final_estimator=LinearRegression() learns how to combine the base predictions.
    final_estimator=LinearRegression(),
    # cv=5 means the meta-features are built from 5-fold cross-validated predictions.
    cv=5,
    # passthrough=False means the final layer uses only the base-model outputs.
    passthrough=False,
    n_jobs=1,
)
stacking_model.fit(X_train, y_train)
predictions = stacking_model.predict(X_test)

print("StackingRegressor example")
print("Test R^2:", round(r2_score(y_test, predictions), 3))
print("Test MSE:", round(mean_squared_error(y_test, predictions), 2))
print(
    "Interpretation: stacking regression learns how to combine different"
    " numeric predictors instead of averaging them blindly."
)
