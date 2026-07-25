from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Preprocessing Pipeline With A Model
# ============================================================
#
# What this example shows:
# - how to split preprocessing for numeric and categorical columns
# - how to handle missing values with SimpleImputer
# - how to standardize numeric features
# - how to one-hot encode categorical features
# - how to attach a model at the end of the pipeline
#
# Why we use a pipeline:
# - it keeps preprocessing and modeling in one reproducible workflow
# - it prevents data leakage because each step is fit only on training data
# - it makes train/predict code cleaner and easier to reuse


# Load a built-in sklearn dataset.
# The wine dataset is numeric, so we will also derive a few categorical columns
# from it in order to demonstrate mixed-type preprocessing.
wine = load_wine(as_frame=True)
df = wine.frame.copy()

# target is the label we want to predict.
y = df["target"]
X = df.drop(columns=["target"]).copy()


# Create a few categorical columns from numeric measurements.
# This keeps the data sklearn-native while still giving us a realistic
# mixed numeric + categorical preprocessing example.
X["alcohol_band"] = pd.cut(
    X["alcohol"],
    bins=[0, 12.5, 13.5, np.inf],
    labels=["low", "medium", "high"],
)
X["malic_acid_band"] = pd.cut(
    X["malic_acid"],
    bins=[0, 2.0, 3.5, np.inf],
    labels=["low", "medium", "high"],
)

# Optionally drop the numeric columns we just binned so the categorical
# transformation has a clearer purpose in the example.
X = X.drop(columns=["alcohol", "malic_acid"])


# Identify numeric and categorical columns automatically.
# include=np.number selects numeric dtypes.
# include=["object", "category"] selects text/category columns.
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns


# Create some missing values by hand so the imputers have work to do.
# random_state via numpy's Generator makes the missing-value pattern reproducible.
rng = np.random.default_rng(42)
for _ in range(120):
    random_row = rng.choice(X.index)
    random_col = rng.choice(X.columns)
    X.loc[random_row, random_col] = np.nan


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 reserves 25% of the rows for evaluation.
    test_size=0.25,
    # random_state=0 makes the split reproducible.
    random_state=0,
    # stratify=y preserves the class balance across train and test.
    stratify=y,
)


# ============================================================
# 1. Create the numeric pipeline
# ============================================================
#
# SimpleImputer(strategy="mean"):
# - fills missing numeric values with the column mean
#
# StandardScaler():
# - centers each numeric feature around 0
# - scales it to roughly unit variance
# - helps models like logistic regression train more reliably
num_vals = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ]
)


# ============================================================
# 2. Create the categorical pipeline
# ============================================================
#
# SimpleImputer(strategy="most_frequent"):
# - fills missing categorical values with the most common category
#
# OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"):
# - turns categories into binary indicator columns
# - drop="first" removes one level per feature to reduce redundancy
# - sparse_output=False returns a dense array, which is easier to inspect
# - handle_unknown="ignore" avoids errors if test data contains unseen levels
cat_vals = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ohe",
            OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore",
            ),
        ),
    ]
)


# ============================================================
# 3. Combine both pipelines with ColumnTransformer
# ============================================================
#
# ColumnTransformer applies:
# - the numeric pipeline only to numeric columns
# - the categorical pipeline only to categorical columns
#
# This is the cleanest way to preprocess mixed feature types.
preprocess = ColumnTransformer(
    transformers=[
        ("num_preprocess", num_vals, num_cols),
        ("cat_preprocess", cat_vals, cat_cols),
    ]
)


# ============================================================
# 4. Add a model at the end of the pipeline
# ============================================================
#
# LogisticRegression is a simple, strong baseline classifier.
#
# max_iter=5000:
# - gives the optimizer enough iterations to converge
#
# random_state=0:
# - makes stochastic behavior reproducible if the solver uses it
full_pipeline = Pipeline(
    [
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=5000, random_state=0)),
    ]
)


# Fit the entire workflow on training data only.
# This means imputers, scaler, encoder, and model all learn from X_train.
full_pipeline.fit(X_train, y_train)


# Predict class labels for the held-out test data.
y_pred = full_pipeline.predict(X_test)


# Also inspect the pure preprocessing output for learning purposes.
# named_steps lets us access steps inside the pipeline by name.
x_transformed = full_pipeline.named_steps["preprocess"].transform(X_test)

print("Preprocessing pipeline with model example")
print("\nNumeric columns:")
print(list(num_cols))

print("\nCategorical columns:")
print(list(cat_cols))

print("\nTransformed test matrix shape:")
print(x_transformed.shape)

print("\nAccuracy:")
print(round(accuracy_score(y_test, y_pred), 3))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

print(
    "Interpretation: the pipeline handles missing values, scaling, encoding,"
    " and classification in one object. That makes preprocessing reproducible"
    " and keeps the model workflow cleaner."
)
