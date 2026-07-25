from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ============================================================
# Support Vector Machine With A Linear Kernel
# ============================================================
#
# What this method does:
# - finds a separating boundary between classes
# - maximizes the margin around that boundary
# - uses a straight-line separator in the scaled feature space
#
# Why we use it:
# - it performs well when classes are close to linearly separable
# - the margin idea gives a strong geometric intuition
# - it is a useful baseline before trying more flexible kernels


# ============================================================
# Build paths to the dataset and output image
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "svm_linear_students.csv"
OUTPUT_PATH = ROOT_DIR / "supervised" / "SVM" / "output" / "svm_linear_kernel_plot.png"


# ============================================================
# Load the dataset
# ============================================================

# This dataset contains two numeric features:
# 1. hours_studied
# 2. practice_projects
#
# The target column is result:
# - pass
# - fail
df = pd.read_csv(DATA_PATH)

print("First 5 rows:")
print(df.head())

print("\nClass counts:")
print(df["result"].value_counts())


# ============================================================
# Prepare features and labels
# ============================================================

X = df[["hours_studied", "practice_projects"]]
y = df["result"]


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

# SVM usually works better when features are on a similar scale.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# Train an SVM with a linear kernel
# ============================================================

# A linear kernel tries to separate the classes with a straight line.
model = SVC(kernel="linear")
model.fit(X_train_scaled, y_train)


# ============================================================
# Evaluate the model
# ============================================================

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=["fail", "pass"]))
print("\nSupport vectors per class:", model.n_support_)


# ============================================================
# Plot the decision boundary
# ============================================================

# Create a dense grid of points that covers the feature space.
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400),
)

# Predict the class of each grid point so we can color the regions.
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = model.predict(grid_points)
grid_predictions_numeric = np.where(grid_predictions == "pass", 1, 0).reshape(xx.shape)

# Convert class labels into numbers for plotting colors.
y_train_numeric = np.where(y_train.to_numpy() == "pass", 1, 0)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, grid_predictions_numeric, alpha=0.28, cmap="coolwarm")
plt.scatter(
    X_train_scaled[:, 0],
    X_train_scaled[:, 1],
    c=y_train_numeric,
    cmap="coolwarm",
    edgecolors="black",
    s=55,
    alpha=0.9,
)

# Highlight the support vectors because they define the margin.
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=170,
    facecolors="none",
    edgecolors="black",
    linewidths=1.5,
    label="Support vectors",
)

plt.title("SVM with Linear Kernel")
plt.xlabel("hours_studied (scaled)")
plt.ylabel("practice_projects (scaled)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
plt.close()

print("\nPlot saved to:")
print(OUTPUT_PATH)
