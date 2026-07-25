from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Hyperparameter Tuning
# ============================================================
#
# Hyperparameters are choices we make before training the model.
# Examples:
# - number of neighbors in KNN
# - max_depth in a decision tree
# - alpha in Ridge or Lasso
#
# We do not "learn" hyperparameters directly from fit().
# Instead, we search over candidate values and compare scores.
#
# Why we use it:
# - one arbitrary setting can be misleading
# - different datasets need different model complexity
# - cross-validation gives a more reliable signal than one split alone


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
    # stratify=y preserves the class balance.
    stratify=y,
)

# Pipeline keeps preprocessing and modeling together in one object.
# That matters during cross-validation because scaling must be fit
# only on each training fold, not on the full dataset in advance.
pipeline = Pipeline(
    # "scaler" standardizes each fold's training features.
    # "model" is the estimator whose hyperparameters we are tuning.
    [
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ]
)

# This grid lists the settings we want to test.
# n_neighbors controls model flexibility.
# weights decides whether all neighbors count equally.
# p controls the distance metric:
# - p=1 -> Manhattan distance
# - p=2 -> Euclidean distance
param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 15],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
}

# GridSearchCV trains and evaluates one model for every parameter
# combination in the grid. It uses cross-validation to score them.
# n_jobs=1 avoids Windows sandbox issues with parallel worker creation.
grid_search = GridSearchCV(
    # estimator=pipeline is the object being tuned.
    # param_grid lists every hyperparameter combination to try.
    # cv=5 means 5-fold cross-validation.
    # scoring="accuracy" ranks models by classification accuracy.
    # n_jobs=1 avoids multiprocessing issues in this environment.
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1,
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, predictions)

results = pd.DataFrame(grid_search.cv_results_)[
    ["params", "mean_test_score", "std_test_score", "rank_test_score"]
].sort_values("rank_test_score")

print("Hyperparameter tuning with GridSearchCV")
print(
    "\nWhat hyperparameter tuning does:"
    "\n- tries different model settings"
    "\n- compares them with cross-validation"
    "\n- keeps the setting that generalizes best"
)

print(
    "\nWhy you would use it:"
    "\n- to avoid guessing values like k in KNN"
    "\n- to reduce underfitting or overfitting"
    "\n- to choose settings using evidence instead of intuition alone"
)

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation accuracy:", round(grid_search.best_score_, 3))
print("Test accuracy with tuned model:", round(test_accuracy, 3))

print("\nTop grid-search results:")
print(results.head(8).to_string(index=False))

print("\nClassification report for the tuned model:")
print(classification_report(y_test, predictions, target_names=dataset.target_names))

print(
    "Interpretation: hyperparameter tuning is about finding a better balance"
    " between underfitting and overfitting using cross-validation, instead of"
    " trusting one arbitrary setting."
)
