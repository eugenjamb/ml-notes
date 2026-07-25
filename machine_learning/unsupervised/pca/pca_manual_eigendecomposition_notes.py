import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# ============================================================
# Manual PCA Notes With Eigendecomposition
# ============================================================
#
# What this method does:
# - builds PCA from the correlation matrix step by step
# - extracts eigenvalues and eigenvectors manually
# - shows how explained variance comes from those quantities
#
# Why we use it:
# - it explains the math behind PCA instead of hiding it in one library call
# - it makes scree plots and cumulative variance easier to interpret
# - it connects linear algebra ideas to machine learning practice


# Save charts into PCA/output so the folder works like visual notes.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Step 1:
# Load a real dataset that has several numeric features.
# The Wine dataset is good for PCA because it has many correlated columns.
wine = load_wine()
feature_names = wine.feature_names


# Step 2:
# Put the feature matrix into a DataFrame so column names are easy to work with.
data_matrix = pd.DataFrame(wine.data, columns=feature_names)


# Step 3:
# Standardize the features before PCA-style analysis.
# This makes every feature comparable on the same scale.
scaler = StandardScaler()
data_matrix_standardized = pd.DataFrame(
    scaler.fit_transform(data_matrix),
    columns=feature_names,
)


# Step 4:
# Compute the correlation matrix.
# This tells us how strongly each feature moves with the others.
correlation_matrix = data_matrix_standardized.corr()

print("Correlation matrix:")
print(correlation_matrix.round(2))


# Step 5:
# Draw a heatmap so the correlations are easier to inspect visually.
plt.figure(figsize=(11, 8))
red_blue = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(
    correlation_matrix,
    vmin=-1,
    vmax=1,
    cmap=red_blue,
    center=0,
    square=True,
    linewidths=0.4,
    cbar_kws={"label": "Correlation"},
)
plt.title("Wine Dataset Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "manual_correlation_heatmap.png"), dpi=180)
plt.show()


# Step 6:
# Perform eigendecomposition on the correlation matrix.
# Eigenvectors give the directions of the principal axes.
# Eigenvalues tell us how much variance each principal axis explains.
eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)


# Step 7:
# Sort the eigenvalues from largest to smallest so PC1, PC2, ... are ordered.
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

print("\nEigenvalues:")
print(np.round(eigenvalues, 4))

print("\nEigenvectors (columns are principal axes):")
print(np.round(eigenvectors, 4))


# Step 8:
# Convert eigenvalues into explained variance proportions.
# This answers: what fraction of the information does each PC keep?
info_prop = eigenvalues / eigenvalues.sum()
cum_info_prop = np.cumsum(info_prop)

print("\nExplained variance ratio:")
print(np.round(info_prop, 4))

print("\nCumulative explained variance ratio:")
print(np.round(cum_info_prop, 4))


# Step 9:
# Draw a scree plot to show how much information each principal axis explains.
pc_numbers = np.arange(1, len(info_prop) + 1)

plt.figure(figsize=(9, 5))
plt.plot(pc_numbers, info_prop, marker="o", linewidth=2.2, color="royalblue")
plt.bar(pc_numbers, info_prop, alpha=0.25, color="skyblue")
plt.title("Scree Plot For Manual PCA")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.xticks(pc_numbers)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "manual_scree_plot.png"), dpi=180)
plt.show()


# Step 10:
# Draw the cumulative information curve.
# This helps choose how many principal components we need.
components_for_95 = np.argmax(cum_info_prop >= 0.95) + 1

plt.figure(figsize=(9, 5))
plt.plot(pc_numbers, cum_info_prop, marker="o", linewidth=2.2, color="darkgreen")
plt.axhline(y=0.95, color="crimson", linestyle="--", label="95% target")
plt.axvline(
    x=components_for_95,
    color="orange",
    linestyle="--",
    label=f"{components_for_95} components",
)
plt.title("Cumulative Explained Variance")
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.xticks(pc_numbers)
plt.ylim(0, 1.05)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "manual_cumulative_variance.png"), dpi=180)
plt.show()

print(f"\nComponents needed to keep at least 95% variance: {components_for_95}")
