from collections import Counter

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


# ============================================================
# Random Forest, Bootstrapping, And Bagging
# ============================================================
#
# This file ties together the main ideas behind random forests:
# 1. Bootstrapping
# 2. Bagging
# 3. Random feature selection
# 4. Training and predicting with scikit-learn
# 5. RandomForestRegressor
#
# Why this belongs in the decision_tree folder:
# - a random forest is built from many decision trees
# - each tree sees a different sample of the training data
# - the forest combines their predictions into one stronger model
#
# scikit-learn notes from the current docs:
# - RandomForestClassifier is a meta-estimator that fits many decision trees
#   on sub-samples of the dataset and averages predictions to improve accuracy
#   and control overfitting.
# - RandomForestClassifier and RandomForestRegressor both use
#   `bootstrap=True` by default, which means each tree is trained on a
#   bootstrap sample instead of the full training set.
# - `max_features` controls how many features are considered at each split,
#   which is the "random feature selection" part of random forests.


# ============================================================
# Part 1: Bootstrapping
# ============================================================
#
# Bootstrapping means sampling from the training data WITH replacement.
# That means:
# - some rows appear multiple times in one bootstrap sample
# - some rows are not selected at all for that tree
# - the omitted rows are called out-of-bag examples for that tree
#
# This creates variation across trees even before we change the features.


cancer = load_breast_cancer()
X_classification = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_classification = pd.Series(cancer.target)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_classification,
    y_classification,
    # test_size=0.25 keeps 25% of the rows for final evaluation.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
    # stratify=y_classification preserves the class ratio in both splits.
    stratify=y_classification,
)

# Create one manual bootstrap sample from the training set.
# resample(..., replace=True) is the key step that makes it bootstrapping.
bootstrap_indices = resample(
    np.arange(len(X_train_cls)),
    replace=True,
    n_samples=len(X_train_cls),
    random_state=42,
)

bootstrap_counts = Counter(bootstrap_indices)
unique_bootstrap_rows = len(bootstrap_counts)
repeated_rows = sum(1 for count in bootstrap_counts.values() if count > 1)
out_of_bag_rows = len(X_train_cls) - unique_bootstrap_rows

print("Bootstrapping example")
print("Training rows:", len(X_train_cls))
print("Rows drawn in the bootstrap sample:", len(bootstrap_indices))
print("Unique rows included:", unique_bootstrap_rows)
print("Rows repeated at least once:", repeated_rows)
print("Rows left out (out-of-bag for this sample):", out_of_bag_rows)


# ============================================================
# Part 2: Bagging
# ============================================================
#
# Bagging stands for bootstrap aggregating.
# The idea is:
# - build many models on many different bootstrap samples
# - let each model make a prediction
# - combine the predictions at the end
#
# For classification:
# - combine with majority vote
#
# For regression:
# - combine with averaging
#
# Bagging reduces variance because one unstable tree no longer controls
# the final answer by itself.


# Start with one plain decision tree as a baseline.
# random_state=42 reproduces the tree structure.
# max_depth=5 limits tree complexity to reduce overfitting.
single_tree = DecisionTreeClassifier(random_state=42, max_depth=5)
single_tree.fit(X_train_cls, y_train_cls)
single_tree_predictions = single_tree.predict(X_test_cls)


# BaggingClassifier in scikit-learn automates the idea above.
# Each tree is trained on a bootstrap sample of the training set.
# estimator=... chooses the base tree used in each bagged model.
# n_estimators=40 means 40 trees are aggregated.
# bootstrap=True enables sampling with replacement.
# random_state=42 fixes the bootstrap draws.
# n_jobs=1 avoids multiprocessing issues here.
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
    n_estimators=40,
    bootstrap=True,
    random_state=42,
    n_jobs=1,
)
bagging_model.fit(X_train_cls, y_train_cls)
bagging_predictions = bagging_model.predict(X_test_cls)

print("\nBagging example")
print("Single decision tree accuracy:", round(accuracy_score(y_test_cls, single_tree_predictions), 3))
print("Bagging classifier accuracy:", round(accuracy_score(y_test_cls, bagging_predictions), 3))


# ============================================================
# Part 3: Random Feature Selection
# ============================================================
#
# Bagging alone changes the training rows for each tree.
# Random forests add one more source of randomness:
# - at each split, each tree looks at only a random subset of features
#
# Why this matters:
# - if one feature is extremely strong, plain bagging trees may keep choosing it
# - random feature selection forces trees to explore different split paths
# - that makes the trees less correlated with one another
#
# Less correlation between trees usually makes the final ensemble stronger.


# In RandomForestClassifier:
# - bootstrap=True handles the row sampling
# - max_features="sqrt" handles the random feature subset at each split
# n_estimators=80 means the forest trains 80 trees.
# max_depth=5 limits how deep each tree can grow.
# oob_score=True estimates validation performance from out-of-bag samples.
forest_model = RandomForestClassifier(
    n_estimators=80,
    max_depth=5,
    bootstrap=True,
    oob_score=True,
    max_features="sqrt",
    random_state=42,
    n_jobs=1,
)
forest_model.fit(X_train_cls, y_train_cls)
forest_predictions = forest_model.predict(X_test_cls)

print("\nRandom feature selection + random forest")
print("Random forest test accuracy:", round(accuracy_score(y_test_cls, forest_predictions), 3))
print("Random forest OOB score:", round(forest_model.oob_score_, 3))
print("Number of trees in the forest:", len(forest_model.estimators_))
print("max_features setting used:", forest_model.max_features)


# ============================================================
# Part 4: Train And Predict Using scikit-learn
# ============================================================
#
# The core sklearn workflow is still the same:
# - create the model object
# - call fit() on training data
# - call predict() on test data or new data
#
# The difference is that the random forest does this internally with
# many trees and then combines their outputs.


new_patients = X_test_cls.iloc[:3]
new_patient_predictions = forest_model.predict(new_patients)
new_patient_probabilities = forest_model.predict_proba(new_patients)

print("\nTrain and predict using scikit-learn")
print("Predictions for 3 held-out patients:", new_patient_predictions.tolist())
print("Prediction probabilities:")
print(np.round(new_patient_probabilities, 3))


# ============================================================
# Part 5: RandomForestRegressor
# ============================================================
#
# Random forests are not only for classification.
# RandomForestRegressor uses the same ensemble idea for numeric targets.
#
# Main difference:
# - each tree predicts a number
# - the forest averages those numbers across trees
#
# This is often useful when one decision tree is too unstable for regression.


diabetes = load_diabetes()
X_regression = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y_regression = pd.Series(diabetes.target)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regression,
    y_regression,
    # test_size=0.25 reserves 25% of the regression rows for testing.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
)

forest_regressor = RandomForestRegressor(
    # n_estimators=120 means 120 regression trees are averaged together.
    # max_depth=6 limits tree complexity.
    # bootstrap=True uses bootstrap samples for each tree.
    # oob_score=True estimates performance from out-of-bag rows.
    # max_features=0.6 means each split sees 60% of the features at random.
    # random_state=42 fixes sampling and tree randomness.
    # n_jobs=1 avoids parallel worker issues in this environment.
    n_estimators=120,
    max_depth=6,
    bootstrap=True,
    oob_score=True,
    max_features=0.6,
    random_state=42,
    n_jobs=1,
)
forest_regressor.fit(X_train_reg, y_train_reg)
regression_predictions = forest_regressor.predict(X_test_reg)

print("\nRandomForestRegressor example")
print("Test R^2:", round(r2_score(y_test_reg, regression_predictions), 3))
print("Test MSE:", round(mean_squared_error(y_test_reg, regression_predictions), 2))
print("OOB score:", round(forest_regressor.oob_score_, 3))


# Feature importances estimate how useful each column was across the forest.
# This is not perfect causal interpretation, but it is a helpful summary.
feature_importance_table = pd.DataFrame(
    {
        "feature": X_regression.columns,
        "importance": np.round(forest_regressor.feature_importances_, 4),
    }
).sort_values("importance", ascending=False)

print("\nTop regression feature importances:")
print(feature_importance_table.head(6).to_string(index=False))


print(
    "\nInterpretation:"
    "\n- Bootstrapping creates different training samples for different trees."
    "\n- Bagging combines many trees to reduce variance."
    "\n- Random forests add random feature selection so the trees become less similar."
    "\n- RandomForestClassifier votes across trees."
    "\n- RandomForestRegressor averages numeric predictions across trees."
)
