import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# ============================================================
# K-Means Clustering
# ============================================================
#
# What this method does:
# - assigns points to the nearest centroid
# - moves centroids to the average of their assigned points
# - repeats until the clusters stabilize
#
# Why we use it:
# - it is one of the most common unsupervised learning algorithms
# - it gives a clear introduction to clustering
# - it shows how unlabeled data can still be grouped into structure

# Load the Iris dataset.
# Each row is one flower, and each column is a measurement.
iris = datasets.load_iris()
samples = iris.data

# We only use the first two features so the clusters are easy to plot.
# column 0 = sepal length
# column 1 = sepal width
x = samples[:, 0]
y = samples[:, 1]
sepal_length_width = np.column_stack((x, y))

# K is the number of clusters we want to build.
k = 3

# Set a seed so the random starting centroids are reproducible.
np.random.seed(42)

# Step 1: Place K random centroids inside the data range.
centroids = np.column_stack(
    (
        np.random.uniform(x.min(), x.max(), size=k),
        np.random.uniform(y.min(), y.max(), size=k),
    )
)


def distance(a, b):
    # Euclidean distance between two 2D points.
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# Each sample will be assigned a cluster label: 0, 1, or 2.
labels = np.zeros(len(sepal_length_width), dtype=int)

# We start with a large error so the loop runs at least once.
error = np.inf

# Repeat until the centroids stop moving.
while error > 0:
    # Step 2: Assign each sample to the nearest centroid.
    for i, point in enumerate(sepal_length_width):
        distances = np.array([distance(point, centroid) for centroid in centroids])
        labels[i] = np.argmin(distances)

    # Step 3: Save the old centroids before updating them.
    centroids_old = centroids.copy()

    # Step 4: Move each centroid to the mean of the points in its cluster.
    for i in range(k):
        points = sepal_length_width[labels == i]

        # Only update the centroid if the cluster has points.
        # Empty clusters can happen when a centroid gets no assignments.
        if len(points) > 0:
            centroids[i] = points.mean(axis=0)

    # Step 5: Measure total centroid movement.
    # When this becomes 0, K-means has converged.
    error = np.linalg.norm(centroids - centroids_old)


# Plot setup.
plt.figure(figsize=(8, 6))
colors = ["tomato", "royalblue", "mediumseagreen"]

# Plot each cluster separately so each group clearly gets one color.
for i in range(k):
    points = sepal_length_width[labels == i]
    if len(points) > 0:
        plt.scatter(
            points[:, 0],
            points[:, 1],
            color=colors[i],
            alpha=0.65,
            s=55,
            label=f"Cluster {i + 1}",
        )

# Plot the final centroids as large black X markers.
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color="black",
    marker="x",
    s=220,
    linewidths=3,
    label="Centroids",
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering on Iris Sepal Data")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

# Example: predict the cluster for brand-new flower measurements.
# These are new [sepal_length, sepal_width] values that were not in training.
new_samples = np.array(
    [
        [5.0, 3.5],
        [6.7, 3.1],
        [5.8, 2.7],
    ]
)

predicted_labels = []

for sample in new_samples:
    # Compare the new sample against every learned centroid.
    distances = np.array([distance(sample, centroid) for centroid in centroids])

    # The closest centroid becomes the predicted cluster.
    predicted_cluster = np.argmin(distances)
    predicted_labels.append(predicted_cluster)

predicted_labels = np.array(predicted_labels)

print("New samples:")
print(new_samples)
print("Predicted cluster labels:")
print(predicted_labels)

# Optional: show the new samples on top of the finished clustering plot.
plt.figure(figsize=(8, 6))

for i in range(k):
    points = sepal_length_width[labels == i]
    if len(points) > 0:
        plt.scatter(
            points[:, 0],
            points[:, 1],
            color=colors[i],
            alpha=0.45,
            s=55,
            label=f"Cluster {i + 1}",
        )

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color="black",
    marker="x",
    s=220,
    linewidths=3,
    label="Centroids",
)

for i in range(len(new_samples)):
    plt.scatter(
        new_samples[i, 0],
        new_samples[i, 1],
        color=colors[predicted_labels[i]],
        edgecolors="black",
        marker="D",
        s=120,
        label="New Sample" if i == 0 else None,
    )

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Predicting New Samples with Learned Centroids")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
