from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# K-Nearest Neighbors (KNN) Classifier
# ============================================================
#
# What this method does:
# - stores the labeled training examples
# - finds the k closest examples to a new point
# - predicts the majority class among those neighbors
#
# Why we use it:
# - it is easy to understand and explain
# - it works well when similar points usually share the same label
# - it is useful for seeing how the choice of k affects overfitting

breast_cancer_data = load_breast_cancer()

# Split the dataset so we can test how well different k values generalize.
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data,
    breast_cancer_data.target,
    test_size=0.2,
    random_state=100
)

scores = {}

# Try many possible neighbor counts.
# Small k can overfit because predictions depend on very local noise.
# Larger k is smoother, but too large can underfit.
k_list = range(1, 101)

accuracies = []

for i in k_list:
    # n_neighbors=i means "look at the i closest training points".
    classifier = KNeighborsClassifier(n_neighbors=i)

    classifier.fit(training_data, training_labels)

    score = classifier.score(validation_data, validation_labels)

    scores[i] = score          
    accuracies.append(score)   

largest_score = max(scores, key=scores.get)

print("Best fit K:", largest_score)
print("Score:", scores[largest_score])

# Plot validation accuracy so we can see which k balances bias and variance best.
plt.plot(k_list, accuracies)

# Label the graph so the tradeoff is easy to read.
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")

plt.show()
