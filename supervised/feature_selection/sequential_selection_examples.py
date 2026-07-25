from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as MlxtendSFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "examples" / "wrapper_methods_student_performance.csv"

# ============================================================
# Sequential Forward Selection and Backward Selection
# ============================================================
#
# Wrapper methods use a model again and again while testing
# different feature subsets.
#
# Forward selection:
# - Start with zero features.
# - Add the best next feature at each step.
#
# Backward selection:
# - Start with all features.
# - Remove the least useful feature at each step.
#
# Floating selection:
# - Like forward/backward selection, but it can also undo
#   one step if that improves the score.
#
# Why we use these methods:
# - they search feature subsets more deliberately than random guessing
# - they help find smaller sets of predictors that still perform well
# - they show the tradeoff between exhaustive search and practical runtime

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

base_model = LogisticRegression(max_iter=1000, random_state=42)


def subset_score(selected_features):
    """Score one feature subset with cross-validation on the training set."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    scores = cross_val_score(
        pipeline,
        X_train[selected_features],
        y_train,
        cv=5,
        scoring="accuracy",
    )
    return scores.mean()


print("Manual idea of Sequential Forward Selection")
remaining_features = feature_columns.copy()
selected_features = []

# We stop after 4 picks so the example stays easy to read.
for step in range(4):
    best_feature = None
    best_score = -1

    for feature in remaining_features:
        current_subset = selected_features + [feature]
        current_score = subset_score(current_subset)

        if current_score > best_score:
            best_score = current_score
            best_feature = feature

    selected_features.append(best_feature)
    remaining_features.remove(best_feature)

    print(
        f"Step {step + 1}: added '{best_feature}' "
        f"-> CV accuracy = {best_score:.3f}"
    )

print("\nManual forward-selection subset:", selected_features)

# ============================================================
# mlxtend Sequential Feature Selection
# ============================================================

scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train)
scaled_test = scaler.transform(X_test)


def evaluate_mlxtend_selector(name, selector):
    """Fit one mlxtend selector and evaluate its chosen subset."""
    selector.fit(scaled_train, y_train)

    selected_index_positions = list(selector.k_feature_idx_)
    selected_feature_names = [feature_columns[i] for i in selected_index_positions]

    final_model = LogisticRegression(max_iter=1000, random_state=42)
    final_model.fit(scaled_train[:, selected_index_positions], y_train)
    predictions = final_model.predict(scaled_test[:, selected_index_positions])
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{name}")
    print("Selected features:", selected_feature_names)
    print("Cross-validation accuracy:", round(selector.k_score_, 3))
    print("Test accuracy:", round(accuracy, 3))
    print("subsets_:")

    for subset_size, subset_info in selector.subsets_.items():
        subset_feature_names = [
            feature_columns[i] for i in subset_info["feature_idx"]
        ]
        print(
            f"  {subset_size} features -> {subset_feature_names}, "
            f"avg_score={subset_info['avg_score']:.3f}"
        )

    # This plot shows how the cross-validation accuracy changes
    # as the selector keeps more or fewer features.
    plt.figure(figsize=(8, 5))
    plot_sfs(selector.get_metric_dict(), kind="std_dev")
    plt.title(name)
    plt.xlabel("Number of features")
    plt.ylabel("Cross-validation accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


mlxtend_forward = MlxtendSFS(
    base_model,
    k_features=4,
    forward=True,
    floating=False,
    scoring="accuracy",
    cv=5,
)

mlxtend_backward = MlxtendSFS(
    base_model,
    k_features=4,
    forward=False,
    floating=False,
    scoring="accuracy",
    cv=5,
)

mlxtend_forward_floating = MlxtendSFS(
    base_model,
    k_features=4,
    forward=True,
    floating=True,
    scoring="accuracy",
    cv=5,
)

mlxtend_backward_floating = MlxtendSFS(
    base_model,
    k_features=4,
    forward=False,
    floating=True,
    scoring="accuracy",
    cv=5,
)

evaluate_mlxtend_selector(
    "mlxtend Sequential Forward Selection",
    mlxtend_forward,
)
evaluate_mlxtend_selector(
    "mlxtend Sequential Backward Selection",
    mlxtend_backward,
)
evaluate_mlxtend_selector(
    "mlxtend Sequential Forward Floating Selection",
    mlxtend_forward_floating,
)
evaluate_mlxtend_selector(
    "mlxtend Sequential Backward Floating Selection",
    mlxtend_backward_floating,
)
