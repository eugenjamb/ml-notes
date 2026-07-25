import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# ============================================================
# Decision Tree Classification
# ============================================================
#
# What this method does:
# - repeatedly splits the data using feature-based questions
# - builds a tree of decisions from root to leaf
# - predicts a class from the final leaf reached by each example
#
# Why we use it:
# - trees are easy to visualize and explain
# - they can handle non-linear decision boundaries
# - they make concepts like depth and pruning very concrete

# ============================================================
# Load and inspect the dataset
# ============================================================

# Dataset:
# https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data

cols = [
    'name','landmass','zone','area','population','language',
    'religion','bars','stripes','colours','red','green','blue',
    'gold','white','black','orange','mainhue','circles',
    'crosses','saltires','quarters','sunstars','crescent',
    'triangle','icon','animate','text','topleft','botright'
]

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data",
    names=cols
)

# Features to use as predictors
var = [
    'red','green','blue','gold','white','black','orange',
    'mainhue','bars','stripes','circles','crosses',
    'saltires','quarters','sunstars','triangle','animate'
]

# Number of countries in each continent
print(df['landmass'].value_counts())

# ============================================================
# Explore Europe and Oceania flags
# ============================================================

# Filter rows for Europe (3) and Oceania (6)
df_36 = df[df['landmass'].isin([3, 6])]

print(df_36.head())

# Compare average feature values by continent
print(df.groupby('landmass')[var].mean())

# ============================================================
# Prepare labels and feature matrix
# ============================================================

# Create binary labels
# 1 = Europe or Oceania
# 0 = All other continents
labels = (df["landmass"].isin([3,6])) * 1

print(labels)

# Convert categorical variables into dummy variables
data = pd.get_dummies(df[var])

# ============================================================
# Split data into training and testing sets
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    random_state=1,
    test_size=0.4
)

# ============================================================
# Tune max_depth
# ============================================================

# Evaluate tree performance for depths 1-20
depths = range(1, 21)
acc_depth = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth)

    dt.fit(x_train, y_train)

    acc_depth.append(dt.score(x_test, y_test))

print(acc_depth)

# Visualize model accuracy by tree depth
plt.plot(depths, acc_depth)
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.show()

# Best accuracy achieved
max_accuracy = np.max(acc_depth)
print(max_accuracy)

# Depth corresponding to the highest accuracy
best_depth = depths[acc_depth.index(max(acc_depth))]
print(best_depth)

# ============================================================
# Train and visualize the best depth tree
# ============================================================

dt = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=1
)

dt.fit(x_train, y_train)

plt.figure(figsize=(14, 8))

tree.plot_tree(
    dt,
    feature_names=data.columns,
    class_names=['Europe', 'Oceania'],
    filled=True
)

plt.show()

# ============================================================
# Tune pruning with ccp_alpha
# ============================================================

# Cost-complexity pruning values
ccp = np.arange(0, 0.05, 0.002)

acc_pruned = []

# Evaluate accuracy for different pruning strengths
for alpha in ccp:

    dt = DecisionTreeClassifier(
        max_depth=best_depth,
        ccp_alpha=alpha,
        random_state=1
    )

    dt.fit(x_train, y_train)

    acc_pruned.append(dt.score(x_test, y_test))

# Visualize pruning performance
plt.plot(ccp, acc_pruned)
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.show()

# Best pruning value
max_pruned_accuracy = max(acc_pruned)

best_ccp = ccp[acc_pruned.index(max_pruned_accuracy)]

print(max_pruned_accuracy)
print(best_ccp)

# ============================================================
# Train final pruned model
# ============================================================

final_tree = DecisionTreeClassifier(
    max_depth=best_depth,
    ccp_alpha=best_ccp,
    random_state=1
)

final_tree.fit(x_train, y_train)

# ============================================================
# Visualize final pruned decision tree
# ============================================================

plt.figure(figsize=(14, 8))

tree.plot_tree(
    final_tree,
    feature_names=data.columns,
    class_names=['Europe', 'Oceania'],
    filled=True
)

plt.show()
