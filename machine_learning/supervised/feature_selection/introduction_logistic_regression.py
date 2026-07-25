from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "examples" / "wrapper_methods_student_performance.csv"

# ============================================================
# Introduction to wrapper methods
# ============================================================
#
# Wrapper methods try different feature combinations and keep
# the subset that helps the model perform best.
#
# In this example:
# 1. We load our own small dataset.
# 2. We train a Logistic Regression model.
# 3. We inspect the baseline result before feature selection.
#
# Later files will use this same model inside forward selection,
# backward selection, floating selection, and RFE.
#
# Why this matters:
# - wrapper methods need a base model to score feature subsets
# - logistic regression is a good choice because it is fast and interpretable
# - this baseline gives us something to compare feature-selection results against

data = pd.read_csv(DATA_PATH)

print("Dataset preview:")
print(data.head())

feature_columns = [
    "study_hours",
    "attendance",
    "practice_tests",
    "sleep_hours",
    "social_media_hours",
    "gaming_hours",
    "coffee_cups",
    "group_study",
    "assignments_completed",
]

X = data[feature_columns]
y = data["passed_exam"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression is a common model for wrapper-method lessons
# because it is simple, fast, and easy to interpret.
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)

print("\nBaseline model with all features")
print("Training accuracy:", round(accuracy_score(y_train, train_predictions), 3))
print("Testing accuracy:", round(accuracy_score(y_test, test_predictions), 3))

# The coefficients show how strongly each feature affects the prediction.
coefficients = pd.DataFrame(
    {
        "feature": feature_columns,
        "coefficient": model.coef_[0],
    }
).sort_values("coefficient", key=abs, ascending=False)

print("\nLogistic Regression coefficients")
print(coefficients)

print(
    "\nWhy this matters: wrapper methods will now test smaller subsets of these"
    " features to see whether we can keep the important information and remove"
    " weaker or noisy columns."
)
