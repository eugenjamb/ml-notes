from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ============================================================
# Support Vector Machine With A Polynomial Kernel
# ============================================================
#
# What this method does:
# - keeps the same SVM margin idea
# - uses a polynomial transformation so the boundary can bend
# - handles patterns that a straight line cannot separate well
#
# Why we use it:
# - XOR-style data is a classic example where linear separation fails
# - it shows how kernels add flexibility without manually engineering features
# - it helps compare simple and more complex decision boundaries


# ============================================================
# Build paths to the dataset and output image
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "svm_polynomial_xor.csv"
OUTPUT_PATH = ROOT_DIR / "supervised" / "SVM" / "output" / "svm_polynomial_kernel_plot.png"


# ============================================================
# Load the dataset
# ============================================================

# This dataset follows an XOR-like pattern.
# Polynomial kernels are useful when the boundary bends in a way
# that a straight line cannot represent.
df = pd.read_csv(DATA_PATH)

print("First 5 rows:")
print(df.head())

print("\nClass counts:")
print(df["pattern"].value_counts())


# ============================================================
# Prepare features and labels
# ============================================================

X = df[["feature_x", "feature_y"]]
y = df["pattern"]


# ============================================================
# Split the data into training and testing sets
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# Scale the features
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# Train an SVM with a polynomial kernel
# ============================================================

# degree=2 works well for this XOR-style pattern
# because it can create a more complex curved separator.
model = SVC(kernel="poly", degree=2, coef0=1)
model.fit(X_train_scaled, y_train)


# ============================================================
# Evaluate the model
# ============================================================

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=["different_sign", "same_sign"]))
print("\nSupport vectors per class:", model.n_support_)


# ============================================================
# Plot the polynomial decision regions
# ============================================================

x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500),
)

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = model.predict(grid_points)
grid_predictions_numeric = np.where(grid_predictions == "same_sign", 1, 0).reshape(xx.shape)

y_train_numeric = np.where(y_train.to_numpy() == "same_sign", 1, 0)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, grid_predictions_numeric, alpha=0.28, cmap="plasma")
plt.scatter(
    X_train_scaled[:, 0],
    X_train_scaled[:, 1],
    c=y_train_numeric,
    cmap="plasma",
    edgecolors="black",
    s=55,
    alpha=0.9,
)
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=170,
    facecolors="none",
    edgecolors="black",
    linewidths=1.5,
    label="Support vectors",
)

plt.title("SVM with Polynomial Kernel")
plt.xlabel("feature_x (scaled)")
plt.ylabel("feature_y (scaled)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
plt.close()

print("\nPlot saved to:")
print(OUTPUT_PATH)
