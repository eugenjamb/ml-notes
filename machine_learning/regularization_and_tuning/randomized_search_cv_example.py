from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Hyperparameter Tuning With RandomizedSearchCV
# ============================================================
#
# RandomizedSearchCV is an alternative to GridSearchCV.
#
# What it does:
# - samples a fixed number of parameter combinations at random
# - evaluates each sampled combination with cross-validation
# - keeps the combination with the best average validation score
#
# Why we use it:
# - it is often faster than checking every possible combination
# - it scales better when there are many hyperparameters
# - it is useful when we want a good answer without an exhaustive search
#
# Key idea:
# - GridSearchCV tries every point in a search grid
# - RandomizedSearchCV tries only n_iter random combinations
# - that usually saves time when the search space is large


dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 reserves 25% of the rows for testing.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
    # stratify=y preserves the class ratio in both sets.
    stratify=y,
)

# Put scaling and the model in one pipeline so cross-validation
# fits preprocessing correctly inside each fold.
pipeline = Pipeline(
    # The pipeline first scales features, then runs KNN.
    # Keeping them together ensures cross-validation handles preprocessing correctly.
    [
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ]
)

# This search space is intentionally larger than the grid-search example.
# Randomized search is most useful when the total number of combinations
# starts getting large enough that exhaustive search is wasteful.
param_distributions = {
    "model__n_neighbors": list(range(1, 31)),
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
    "model__leaf_size": list(range(10, 61, 5)),
}

# RandomizedSearchCV samples n_iter combinations from the search space.
# random_state makes the random sampling reproducible.
# n_jobs=1 avoids Windows sandbox worker issues in this environment.
random_search = RandomizedSearchCV(
    # estimator=pipeline is the object being tuned.
    # param_distributions defines the search space to sample from.
    # n_iter=20 means 20 random combinations will be tested.
    # cv=5 means 5-fold cross-validation.
    # scoring="accuracy" ranks the models by accuracy.
    # random_state=42 makes the random sampling reproducible.
    # n_jobs=1 avoids parallel worker issues here.
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=1,
)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, predictions)

results = pd.DataFrame(random_search.cv_results_)[
    ["params", "mean_test_score", "std_test_score", "rank_test_score"]
].sort_values("rank_test_score")

total_possible_combinations = (
    len(param_distributions["model__n_neighbors"])
    * len(param_distributions["model__weights"])
    * len(param_distributions["model__p"])
    * len(param_distributions["model__leaf_size"])
)

print("Hyperparameter tuning with RandomizedSearchCV")
print(
    "\nWhat RandomizedSearchCV does:"
    "\n- samples random hyperparameter combinations"
    "\n- scores them with cross-validation"
    "\n- returns the best sampled setting"
)

print(
    "\nWhy you would use it:"
    "\n- to search larger spaces more efficiently"
    "\n- to reduce runtime compared with exhaustive grid search"
    "\n- to get strong candidate settings without testing everything"
)

print("\nSearch-space summary:")
print("Total possible combinations in this example:", total_possible_combinations)
print("Random combinations actually tested:", random_search.n_iter)

print("\nBest parameters:")
print(random_search.best_params_)

print("\nBest cross-validation accuracy:", round(random_search.best_score_, 3))
print("Test accuracy with tuned model:", round(test_accuracy, 3))

print("\nTop randomized-search results:")
print(results.head(8).to_string(index=False))

print("\nClassification report for the tuned model:")
print(classification_report(y_test, predictions, target_names=dataset.target_names))

print(
    "Interpretation: RandomizedSearchCV gives up exhaustive coverage in exchange"
    " for speed. That tradeoff is often worth it when the search space is too"
    " large for grid search to be practical."
)
