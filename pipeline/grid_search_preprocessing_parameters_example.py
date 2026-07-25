import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


# ============================================================
# Grid Search Over Preprocessing And Model Parameters
# ============================================================
#
# GridSearchCV can tune more than the final model. It can also select:
# - how numerical missing values are filled
# - which numerical scaler is used
# - how categorical missing values are filled
# - how OneHotEncoder represents categories
# - the model's regularization strength
#
# Searching the complete pipeline is important because every candidate's
# preprocessing is fitted only on its cross-validation training fold. This
# prevents information from validation folds leaking into preprocessing.


# Load an sklearn regression dataset as a pandas DataFrame.
diabetes = load_diabetes(as_frame=True)
X = diabetes.data.copy()
y = diabetes.target.copy()


# The original dataset is entirely numerical. Create two categorical features
# so this example can tune both numerical and categorical preprocessing.
X["bmi_band"] = pd.cut(
    X["bmi"],
    # bins defines the boundaries used to convert BMI into three categories.
    bins=[-np.inf, -0.02, 0.02, np.inf],
    labels=["low", "medium", "high"],
)
X["bp_band"] = pd.cut(
    X["bp"],
    bins=[-np.inf, -0.02, 0.02, np.inf],
    labels=["low", "medium", "high"],
)

# Remove the source columns so their information is represented only by the
# new categorical versions.
X = X.drop(columns=["bmi", "bp"])

num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns


# Add reproducible missing values so GridSearchCV can compare imputers.
rng = np.random.default_rng(42)
for _ in range(180):
    row = rng.choice(X.index)
    column = rng.choice(X.columns)
    X.loc[row, column] = np.nan


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 keeps 25% of the data completely outside GridSearchCV.
    test_size=0.25,
    # random_state makes the train/test split reproducible.
    random_state=0,
)


# ============================================================
# 1. Define the two preprocessing branches
# ============================================================

# These are initial values. GridSearchCV will replace or change them using the
# parameter grid below.
numeric_pipeline = Pipeline(
    steps=[
        # SimpleImputer fills missing numerical values. Its strategy is tuned.
        ("imputer", SimpleImputer(strategy="mean")),
        # The complete scaler object is replaced during the grid search.
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        # The categorical imputation strategy is also tuned.
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ohe",
            OneHotEncoder(
                # handle_unknown="ignore" prevents unseen test categories
                # from raising an error during transform or predict.
                handle_unknown="ignore",
                # sparse_output=False produces a dense array. This dataset is
                # small, so a dense representation is easy to inspect.
                sparse_output=False,
            ),
        ),
    ]
)


# Apply each branch only to the columns it understands.
preprocess = ColumnTransformer(
    transformers=[
        ("num_preprocess", numeric_pipeline, num_cols),
        ("cat_preprocess", categorical_pipeline, cat_cols),
    ]
)


# Ridge is sensitive to feature scale, making the scaler comparison meaningful.
# alpha controls L2 regularization and is tuned along with preprocessing.
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", Ridge()),
    ]
)


# ============================================================
# 2. Define preprocessing and model candidates
# ============================================================
#
# Nested pipeline parameters use names joined by double underscores:
# outer_step__transformer__inner_step__parameter
#
# For example:
# preprocess__num_preprocess__imputer__strategy
# means the `strategy` parameter of the numerical `imputer`.
parameter_grid = {
    # mean uses the average and is sensitive to outliers.
    # median uses the middle value and is more resistant to outliers.
    "preprocess__num_preprocess__imputer__strategy": ["mean", "median"],

    # Supplying estimator objects lets GridSearchCV choose the whole scaler:
    # StandardScaler gives zero mean and unit variance.
    # MinMaxScaler maps each feature into the range [0, 1].
    "preprocess__num_preprocess__scaler": [
        StandardScaler(),
        MinMaxScaler(),
    ],

    # most_frequent inserts the most common category.
    # constant inserts the value supplied by fill_value.
    "preprocess__cat_preprocess__imputer__strategy": [
        "most_frequent",
        "constant",
    ],
    "preprocess__cat_preprocess__imputer__fill_value": ["missing"],

    # drop=None keeps an indicator for every category.
    # drop="first" removes the first category from every encoded feature.
    "preprocess__cat_preprocess__ohe__drop": [None, "first"],

    # alpha is Ridge's L2 regularization strength. Larger values shrink model
    # coefficients more strongly and can reduce overfitting.
    "model__alpha": [0.1, 1.0, 10.0],
}


# ============================================================
# 3. Search every parameter combination
# ============================================================
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    # Negative MSE is used because sklearn's scorers are maximized. A value
    # closer to zero represents a lower, and therefore better, MSE.
    scoring="neg_mean_squared_error",
    # cv=5 evaluates every candidate on five train/validation splits.
    cv=5,
    # n_jobs=1 avoids multiprocessing and keeps this teaching example simple.
    n_jobs=1,
    # refit=True retrains the winning full pipeline on all training rows.
    refit=True,
)

grid_search.fit(X_train, y_train)


# ============================================================
# 4. Inspect the selected preprocessing and model settings
# ============================================================
best_pipeline = grid_search.best_estimator_
best_preprocess = best_pipeline.named_steps["preprocess"]
best_num_pipeline = best_preprocess.named_transformers_["num_preprocess"]
best_cat_pipeline = best_preprocess.named_transformers_["cat_preprocess"]

best_num_imputer = best_num_pipeline.named_steps["imputer"]
best_scaler = best_num_pipeline.named_steps["scaler"]
best_cat_imputer = best_cat_pipeline.named_steps["imputer"]
best_ohe = best_cat_pipeline.named_steps["ohe"]
best_model = best_pipeline.named_steps["model"]

print("Grid search over preprocessing parameters")
print("\nBest parameters selected by GridSearchCV:")
for parameter_name, selected_value in grid_search.best_params_.items():
    print(f"{parameter_name}: {selected_value}")

print("\nSelected numerical imputer:")
print(best_num_imputer)
print("Selected numerical imputer parameters:", best_num_imputer.get_params())

print("\nSelected scaler:")
print(best_scaler)
print("Selected scaler parameters:", best_scaler.get_params())

print("\nSelected categorical imputer:")
print(best_cat_imputer)
print("Selected categorical imputer parameters:", best_cat_imputer.get_params())

print("\nSelected OneHotEncoder:")
print(best_ohe)
print("Selected OneHotEncoder parameters:", best_ohe.get_params())

print("\nSelected Ridge model:")
print(best_model)
print("Selected Ridge parameters:", best_model.get_params())

print("\nBest cross-validation negative MSE:")
print(round(grid_search.best_score_, 3))


# ============================================================
# 5. Predict through the winning complete pipeline
# ============================================================
#
# Predict with best_pipeline, not the extracted Ridge object. The raw test
# frame still needs the winning imputers, scaler, and encoder first.
y_pred = best_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))

print("\nTest-set regression metrics:")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R^2:", round(r2_score(y_test, y_pred), 3))

print(
    "\nInterpretation: GridSearchCV selected one complete workflow, including"
    " the numerical imputer, scaler, categorical imputer, one-hot encoding"
    " behavior, and Ridge regularization strength."
)
