import os

# Force a single-thread setup in this environment.
# HistGradientBoostingClassifier internally parallelizes its binning step,
# which can trigger Windows sandbox permission issues if it tries to create
# more worker infrastructure than the environment allows.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ============================================================
# Histogram Gradient Boosting Classifier Example
# ============================================================
#
# HistGradientBoostingClassifier is a faster modern variation of gradient boosting.
#
# What makes it different:
# - it bins feature values into histograms before splitting
# - this can speed up training, especially on larger datasets
# - it still follows the same stage-by-stage boosting idea
#
# Why we include it:
# - it broadens the boosting section beyond the classic implementation
# - it introduces a practical sklearn boosting model you may use later


dataset = load_wine()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    # test_size=0.25 reserves a quarter of the data for testing.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
    # stratify=y keeps the class ratio stable.
    stratify=y,
)

# learning_rate=0.08 controls the step size of each boosting stage.
# max_depth=4 limits tree complexity inside each stage.
# max_iter=120 is the number of boosting iterations.
# random_state=42 makes the fit reproducible.
model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_depth=4,
    max_iter=120,
    random_state=42,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("HistGradientBoostingClassifier example")
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print(
    "Interpretation: histogram gradient boosting keeps the boosting idea but"
    " uses a more efficient internal representation of feature values."
)
