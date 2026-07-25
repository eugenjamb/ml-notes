import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# PCA Projection Visualization
# ============================================================
#
# What this method does:
# - transforms many original features into a smaller number of principal components
# - projects the data into a lower-dimensional space
# - visualizes whether class structure is still visible after compression
#
# Why we use it:
# - it makes PCA easier to interpret visually
# - it shows that useful structure can remain after dimensionality reduction
# - it is helpful for exploratory analysis before downstream modeling


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Step 1:
# Load a labeled dataset so we can color the PCA plot by class.
wine = load_wine()


# Step 2:
# Create a DataFrame for the numeric features and a Series for the target labels.
data_matrix = pd.DataFrame(wine.data, columns=wine.feature_names)
classes = pd.Series(wine.target).map(dict(enumerate(wine.target_names)))


# Step 3:
# Standardize the data before PCA.
# PCA is based on variance, so scaling matters a lot.
scaler = StandardScaler()
data_matrix_standardized = scaler.fit_transform(data_matrix)


# Step 4:
# Build PCA with 4 principal components.
# This follows the same idea as your example with 4 new features.
# n_components=4 means PCA will keep the first 4 principal axes.
pca = PCA(n_components=4)
data_pcomp = pca.fit_transform(data_matrix_standardized)


# Step 5:
# Put the transformed values into a new DataFrame with clear PC names.
data_pcomp = pd.DataFrame(data_pcomp, columns=["PC1", "PC2", "PC3", "PC4"])
data_pcomp["wine_class"] = classes

print("First rows of the PCA-transformed data:")
print(data_pcomp.head())

print("\nExplained variance ratio for PC1-PC4:")
print(pca.explained_variance_ratio_.round(4))


# Step 6:
# Plot the first two principal components.
# If the classes separate well here, PCA kept useful structure.
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=data_pcomp,
    x="PC1",
    y="PC2",
    hue="wine_class",
    palette="Set2",
    s=90,
    alpha=0.82,
    edgecolor="black",
)
plt.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
plt.axvline(0, color="gray", linewidth=0.8, alpha=0.4)
plt.title("Wine Dataset Projected Onto First Two Principal Components")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
plt.legend(title="Wine Class")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sklearn_pca_projection.png"), dpi=180)
plt.show()
