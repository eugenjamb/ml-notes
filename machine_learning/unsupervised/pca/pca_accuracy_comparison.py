import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ============================================================
# PCA Accuracy Comparison
# ============================================================
#
# What this method does:
# - compresses the original features into principal components
# - trains one classifier on PCA features and another on the full data
# - compares whether lower-dimensional data keeps enough useful information
#
# Why we use it:
# - PCA can reduce dimensionality and noise
# - it helps visualize the tradeoff between simplicity and information loss
# - it connects unsupervised feature extraction to supervised performance


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Step 1:
# Load a classification dataset.
# We will compare model accuracy before and after PCA.
wine = load_wine()
data_matrix = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)


# Step 2:
# Standardize the original features so both PCA and SVM work on balanced scales.
scaler = StandardScaler()
data_matrix_standardized = scaler.fit_transform(data_matrix)


# Step 3:
# Compress the data into 4 principal components.
# These 4 columns are a lower-dimensional summary of the original 13 features.
pca_1 = PCA(n_components=4)
X_pca = pca_1.fit_transform(data_matrix_standardized)


# Step 4:
# Split the PCA features into training and testing sets.
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y,
)


# Step 5:
# Train a linear SVM on the PCA-compressed features.
svc_1 = LinearSVC(random_state=0, tol=1e-5, max_iter=10000)
svc_1.fit(X_train_pca, y_train)
score_1 = svc_1.score(X_test_pca, y_test)

print(f"Score for model with 4 PCA features: {score_1:.4f}")


# Step 6:
# Split the original standardized features using the same random state.
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    data_matrix_standardized,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y,
)


# Step 7:
# Train the same model on all original features.
svc_2 = LinearSVC(random_state=0, max_iter=10000)
svc_2.fit(X_train_full, y_train_full)
score_2 = svc_2.score(X_test_full, y_test_full)

print(f"Score for model with original features: {score_2:.4f}")


# Step 8:
# Show how much variance each of the 4 PCA features keeps.
explained_variance = pca_1.explained_variance_ratio_
print("\nExplained variance ratio for the 4 PCA features:")
print(np.round(explained_variance, 4))


# Step 9:
# Make one small chart to compare the two accuracies directly.
plt.figure(figsize=(8, 5))
model_names = ["4 PCA Features", "Original Features"]
scores = [score_1, score_2]
bars = plt.bar(
    model_names,
    scores,
    color=["steelblue", "darkorange"],
    alpha=0.85,
)
plt.ylim(0.0, 1.05)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison: PCA Features vs Original Features")
plt.grid(axis="y", alpha=0.25)

for bar, score in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        score + 0.02,
        f"{score:.3f}",
        ha="center",
        fontsize=11,
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_vs_original_accuracy.png"), dpi=180)
plt.show()
