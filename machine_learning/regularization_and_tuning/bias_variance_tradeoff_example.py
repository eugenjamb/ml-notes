from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# The Bias-Variance Tradeoff
# ============================================================
#
# High bias:
# - the model is too simple
# - it underfits
# - both training and test scores stay low
#
# High variance:
# - the model is too flexible
# - it overfits training data
# - training score is very high, but test score drops
#
# We usually want the middle ground where the model is flexible
# enough to learn patterns but not so flexible that it memorizes noise.
#
# Why we study this:
# - it explains why "more complex" is not always better
# - it helps us choose model size and hyperparameters
# - it connects directly to regularization and tuning


wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# max_depth is a clean way to control tree complexity:
# - shallow depth -> simpler tree -> more bias
# - deep depth -> more flexible tree -> more variance
depth_results = []

for depth in range(1, 13):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    depth_results.append(
        {
            "max_depth": depth,
            "train_accuracy": accuracy_score(y_train, train_predictions),
            "test_accuracy": accuracy_score(y_test, test_predictions),
        }
    )

results = pd.DataFrame(depth_results)
results["gap"] = results["train_accuracy"] - results["test_accuracy"]

best_row = results.loc[results["test_accuracy"].idxmax()]

print("Bias-variance tradeoff with Decision Trees")
print(
    "\nWhat this example is doing:"
    "\n- training the same model family at different depths"
    "\n- measuring both training and test accuracy"
    "\n- using the train/test gap to spot overfitting"
)

print(
    "\nWhy this matters:"
    "\n- low training and low test accuracy suggests underfitting"
    "\n- very high training accuracy with weaker test accuracy suggests overfitting"
    "\n- the best model is usually somewhere between those extremes"
)

print("\nAccuracy by tree depth:")
print(results.round(3).to_string(index=False))

print("\nBest depth based on test accuracy:")
print(best_row.round(3).to_dict())

print(
    "\nInterpretation: shallow trees usually show higher bias because they miss"
    " important patterns. Very deep trees often show higher variance because"
    " they fit the training data too closely. The best depth is often somewhere"
    " in between."
)


plt.figure(figsize=(9, 5))
plt.plot(results["max_depth"], results["train_accuracy"], marker="o", label="Training accuracy")
plt.plot(results["max_depth"], results["test_accuracy"], marker="o", label="Test accuracy")
plt.xlabel("Decision tree max_depth")
plt.ylabel("Accuracy")
plt.title("Bias-Variance Tradeoff Across Decision Tree Depths")
plt.xticks(results["max_depth"])
plt.ylim(0.6, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bias_variance_tradeoff.png", dpi=180)
plt.close()
