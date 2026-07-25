from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# SVC Regularization Controlled By C
# ============================================================
#
# SVC uses the hyperparameter C to control regularization strength.
# Important idea:
# - small C -> stronger regularization, wider margin, more tolerance for mistakes
# - large C -> weaker regularization, less tolerance for mistakes, tighter fit
#
# What this method does:
# - trains the same SVC model with different C values
# - compares accuracy and the number of support vectors
# - shows how regularization changes flexibility
#
# Why we use it:
# - C is one of the most important SVM hyperparameters
# - it directly affects bias-variance behavior
# - it is a clean example of regularization without coefficient penalties


X, y = make_circles(n_samples=260, noise=0.12, factor=0.4, random_state=42)
X = pd.DataFrame(X, columns=["feature_x", "feature_y"])
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 keeps 25% for evaluation.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
    # stratify=y preserves the class ratio.
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

c_values = [0.05, 0.2, 1.0, 5.0, 20.0]
results = []

for c_value in c_values:
    # kernel="rbf" uses a non-linear radial basis function boundary.
    # C controls regularization strength.
    # gamma="scale" lets sklearn choose a feature-scale-aware kernel width.
    model = SVC(kernel="rbf", C=c_value, gamma="scale")
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    results.append(
        {
            "C": c_value,
            "test_accuracy": round(accuracy_score(y_test, predictions), 3),
            "support_vectors": int(model.support_vectors_.shape[0]),
        }
    )

results_table = pd.DataFrame(results)

print("SVC regularization controlled by C")
print(
    "\nWhat C does:"
    "\n- lower C means stronger regularization"
    "\n- higher C means weaker regularization"
    "\n- changing C changes how strictly the model tries to fit the training data"
)

print(
    "\nWhy support vectors matter:"
    "\n- they are the training points that define the decision boundary"
    "\n- more support vectors often means the boundary depends on more points"
    "\n- regularization changes how many points remain critical"
)

print("\nResults across C values:")
print(results_table.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(results_table["C"], results_table["test_accuracy"], marker="o", linewidth=2)
plt.xscale("log")
plt.xlabel("C (log scale)")
plt.ylabel("Test accuracy")
plt.title("SVC Accuracy Across Regularization Strengths")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "svc_regularization_c_accuracy.png", dpi=180)
plt.close()
