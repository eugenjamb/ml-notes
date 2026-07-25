from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# Logistic Regression With A Custom Threshold
# ============================================================
#
# What this method does:
# - logistic regression predicts class probabilities
# - we usually convert those probabilities into classes with a 0.5 cutoff
# - this example shows what happens when we choose a different threshold
#
# Why we use it:
# - some problems care more about precision than recall, or the reverse
# - the default 0.5 cutoff is convenient but not always best
# - threshold tuning is a simple way to match predictions to business goals

# Pick an alternative probability threshold for the final classification step.
ALTERNATIVE_THRESHOLD = 0.60

# Build the dataset path relative to the project root so the script works from any cwd.
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "codecademyU_2.csv"


def main() -> None:
    # Load the local CSV dataset into a pandas DataFrame.
    codecademy_u = pd.read_csv(DATA_PATH)

    # Select the input features used to predict exam results.
    X = codecademy_u[["hours_studied", "practice_test"]]

    # Select the target column that stores whether the student passed.
    y = codecademy_u["passed_exam"]

    # Standardize the feature columns so they are on a similar scale.
    # Logistic regression often trains more reliably when features are scaled.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the standardized data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=51
    )

    # Create the logistic regression model and train it on the training set.
    cc_lr = LogisticRegression()
    cc_lr.fit(X_train, y_train)

    # Predict the default class labels for the test set.
    y_pred = cc_lr.predict(X_test)
    print("Predicted classes:", y_pred)

    # Print the actual labels from the test set for comparison.
    print("True classes:")
    print(y_test.to_numpy())

    # Generate the probability of passing the exam for each test example.
    # predict_proba() is important because threshold tuning works on probabilities,
    # not directly on the already-thresholded class labels.
    probabilities = cc_lr.predict_proba(X_test)[:, 1]
    print("Predicted pass probabilities:", probabilities)

    # Apply the custom threshold to turn probabilities into class predictions.
    # Raising the threshold usually makes positive predictions harder to trigger.
    alternative_predictions = (probabilities >= ALTERNATIVE_THRESHOLD).astype(int)
    print(f"Predicted classes with threshold {ALTERNATIVE_THRESHOLD}:", alternative_predictions)

    # Print the confusion matrix for the model's default predictions.
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Print the main evaluation metrics for the model's default predictions.
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    # Print the evaluation metrics again using the alternative threshold predictions.
    # Comparing both sets of metrics shows the tradeoff created by the new cutoff.
    print(f"Accuracy at threshold {ALTERNATIVE_THRESHOLD}:", accuracy_score(y_test, alternative_predictions))
    print(f"F1 score at threshold {ALTERNATIVE_THRESHOLD}:", f1_score(y_test, alternative_predictions))
    print(f"Precision at threshold {ALTERNATIVE_THRESHOLD}:", precision_score(y_test, alternative_predictions))
    print(f"Recall at threshold {ALTERNATIVE_THRESHOLD}:", recall_score(y_test, alternative_predictions))


if __name__ == "__main__":
    main()
