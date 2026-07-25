from pathlib import Path

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "examples" / "wrapper_methods_student_performance.csv"

# ============================================================
# Recursive Feature Elimination (RFE)
# ============================================================
#
# RFE starts with all features, trains a model, removes the
# weakest feature, and repeats until only the chosen number
# of features remains.
#
# This is another wrapper-style approach because the model is
# used again and again during the selection process.
#
# Why we use it:
# - it ranks features by how useful they are to the model
# - it can simplify a model without trying every possible subset
# - it is a practical bridge between interpretability and performance

data = pd.read_csv(DATA_PATH)

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

model = LogisticRegression(max_iter=1000, random_state=42)

# Keep 4 features so we can compare them against the full set.
selector = RFE(estimator=model, n_features_to_select=4)
selector.fit(X_train_scaled, y_train)

selected_features = X.columns[selector.support_].tolist()

ranking = pd.DataFrame(
    {
        "feature": feature_columns,
        "rank": selector.ranking_,
        "selected": selector.support_,
    }
).sort_values("rank")

print("RFE ranking")
print(ranking)

print("\nSelected features after RFE:")
print(selected_features)

# Train one model with every feature for comparison.
all_feature_model = LogisticRegression(max_iter=1000, random_state=42)
all_feature_model.fit(X_train_scaled, y_train)
all_feature_predictions = all_feature_model.predict(X_test_scaled)
all_feature_accuracy = accuracy_score(y_test, all_feature_predictions)

# Train a second model with only the RFE-selected features.
selected_model = LogisticRegression(max_iter=1000, random_state=42)
selected_model.fit(X_train_scaled[:, selector.support_], y_train)
selected_predictions = selected_model.predict(X_test_scaled[:, selector.support_])
selected_accuracy = accuracy_score(y_test, selected_predictions)

print("\nEvaluation")
print("Accuracy with all features:", round(all_feature_accuracy, 3))
print("Accuracy with RFE-selected features:", round(selected_accuracy, 3))

print(
    "\nInterpretation: if the RFE model keeps similar accuracy while using fewer"
    " columns, then feature selection helped simplify the model without losing"
    " much predictive power."
)
