import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Grid Search With A Mixed-Type Preprocessing Pipeline
# ============================================================
#
# What this example shows:
# - how to split categorical and numerical preprocessing
# - how to place preprocessing and the model into one pipeline
# - how to search across different regression model families
# - how to inspect the best model after GridSearchCV finishes
#
# Why this matters:
# - preprocessing should be learned only from the training data
# - GridSearchCV should evaluate the entire workflow, not just the raw model
# - swapping models inside one pipeline is a clean pattern for model selection


# Load a built-in sklearn regression dataset.
# We use diabetes because the target is numeric, which makes it suitable
# for LinearRegression, Ridge, and Lasso.
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame.copy()

y = df["target"]
X = df.drop(columns=["target"]).copy()


# The diabetes dataset is numeric, so we derive a couple of categorical
# columns from numeric values to demonstrate mixed preprocessing.
X["bmi_band"] = pd.cut(
    X["bmi"],
    bins=[-np.inf, -0.02, 0.02, np.inf],
    labels=["low", "medium", "high"],
)
X["bp_band"] = pd.cut(
    X["bp"],
    bins=[-np.inf, -0.02, 0.02, np.inf],
    labels=["low", "medium", "high"],
)

# Drop the source columns we binned so the categorical branch is more explicit.
X = X.drop(columns=["bmi", "bp"])


# Detect numeric and categorical columns automatically.
# include=np.number selects numeric columns.
# include=["object", "category"] selects categorical/text-like columns.
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns


# Create missing values by hand so the imputer steps actually do work.
# The random generator keeps the missing-value pattern reproducible.
rng = np.random.default_rng(42)
for _ in range(180):
    random_row = rng.choice(X.index)
    random_col = rng.choice(X.columns)
    X.loc[random_row, random_col] = np.nan


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # random_state=0 makes the split reproducible.
    random_state=0,
    # test_size=0.25 reserves 25% of the rows for evaluation.
    test_size=0.25,
)


# ============================================================
# 1. Build the categorical and numerical pipelines
# ============================================================
#
# Categorical branch:
# - SimpleImputer(strategy="most_frequent") fills missing categories
#   with the most common category in the training data.
# - OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
#   expands categories into binary columns.
#   drop="first" removes one level per feature to reduce redundancy.
#   sparse_output=False returns a dense matrix for easier inspection.
#   handle_unknown="ignore" prevents test-time errors for unseen categories.
cat_vals = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ohe",
            OneHotEncoder(
                sparse_output=False,
                drop="first",
                handle_unknown="ignore",
            ),
        ),
    ]
)


# Numerical branch:
# - SimpleImputer(strategy="mean") fills missing numeric values with the column mean.
# - StandardScaler() centers and scales the numeric features.
num_vals = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ]
)


# ============================================================
# 2. Combine preprocessing with ColumnTransformer
# ============================================================
#
# ColumnTransformer applies:
# - the categorical pipeline only to categorical columns
# - the numerical pipeline only to numerical columns
preprocess = ColumnTransformer(
    transformers=[
        ("cat_preprocess", cat_vals, cat_cols),
        ("num_preprocess", num_vals, num_cols),
    ]
)


# ============================================================
# 3. Create a pipeline with preprocessing + regression
# ============================================================
#
# The pipeline gives us one object that can:
# - fit the imputers and encoder
# - transform the features
# - fit the regression model
# all in the correct order.
pipeline = Pipeline(
    [
        ("preprocess", preprocess),
        ("regr", LinearRegression()),
    ]
)


# ============================================================
# 4. Define the GridSearchCV search space
# ============================================================
#
# Each dictionary describes one model family and its tunable parameters.
#
# LinearRegression:
# - fit_intercept decides whether the model learns an intercept term.
#
# Ridge:
# - alpha controls L2 regularization strength.
#
# Lasso:
# - alpha controls L1 regularization strength.
# - max_iter gives the optimizer enough steps to converge.
search_space = [
    {
        "regr": [LinearRegression()],
        "regr__fit_intercept": [True, False],
    },
    {
        "regr": [Ridge()],
        "regr__alpha": [0.01, 0.1, 1, 10, 100],
    },
    {
        "regr": [Lasso(max_iter=10000, random_state=0)],
        "regr__alpha": [0.01, 0.1, 1, 10, 100],
    },
]


# ============================================================
# 5. Run GridSearchCV
# ============================================================
#
# scoring="neg_mean_squared_error":
# - sklearn expects "higher is better" scores
# - MSE is naturally lower-is-better, so sklearn uses its negative form
#
# cv=5:
# - 5-fold cross-validation
#
# n_jobs=1:
# - avoids multiprocessing issues in this environment
gs = GridSearchCV(
    pipeline,
    search_space,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=1,
)


# Fit the entire search object on the training data.
gs.fit(X_train, y_train)


# ============================================================
# 6. Inspect the best pipeline and best model
# ============================================================
best_pipeline = gs.best_estimator_
best_regression_model = best_pipeline.named_steps["regr"]

print("Grid search regression pipeline example")
print("\nBest pipeline:")
print(best_pipeline)

print("\nBest regression model:")
print(best_regression_model)

print("\nBest GridSearchCV parameters:")
print(gs.best_params_)

print("\nBest cross-validation score (negative MSE):")
print(round(gs.best_score_, 3))


# Show the chosen model's own parameter settings.
best_model_hyperparameters = best_regression_model.get_params()
print("\nHyperparameters of the best regression model:")
print(best_model_hyperparameters)


# Access the hyperparameters of a nested preprocessing step.
cat_preprocess_hyperparameters = (
    best_pipeline.named_steps["preprocess"]
    .named_transformers_["cat_preprocess"]
    .named_steps["imputer"]
    .get_params()
)
print("\nHyperparameters of the categorical imputer:")
print(cat_preprocess_hyperparameters)


# ============================================================
# 7. Use the best pipeline to predict on test data
# ============================================================
#
# Even though we also extracted the best model above, predictions should usually
# go through the best full pipeline so preprocessing happens automatically.
y_pred = best_pipeline.predict(X_test)


# ============================================================
# 8. Display regression metrics
# ============================================================
#
# We show several common regression metrics so the example is easy to compare
# against future models.
mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))

print("\nRegression metrics on the test set:")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("Median absolute error:", round(median_absolute_error(y_test, y_pred), 3))
print("Max error:", round(max_error(y_test, y_pred), 3))
print("Explained variance:", round(explained_variance_score(y_test, y_pred), 3))
print("R^2:", round(r2_score(y_test, y_pred), 3))
print("MAPE:", round(mean_absolute_percentage_error(y_test, y_pred), 3))

print(
    "\nInterpretation: GridSearchCV selected the best preprocessing + model"
    " workflow based on cross-validated negative MSE. After that, we used the"
    " best full pipeline to predict on the test set and evaluate it with common"
    " regression metrics."
)
