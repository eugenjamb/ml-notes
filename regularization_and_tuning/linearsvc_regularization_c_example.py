import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# LinearSVC Regularization Controlled By C
# ============================================================
#
# LinearSVC is the linear-margin version of SVM classification.
# Like SVC, its regularization is mainly controlled by C.
#
# What this method does:
# - fits the same linear SVM model at several C values
# - compares train and test accuracy
# - uses the train/test gap to show how regularization changes fit
#
# Why we use it:
# - LinearSVC is efficient on linear problems and larger feature spaces
# - C is the main knob for controlling complexity
# - it shows that regularization ideas carry across model families


dataset = load_wine()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.25 reserves 25% for evaluation.
    test_size=0.25,
    # random_state=42 makes the split reproducible.
    random_state=42,
    # stratify=y keeps class proportions similar in train and test.
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
results = []

for c_value in c_values:
    # C controls regularization strength.
    # max_iter=10000 gives the optimizer room to converge.
    # dual=False is often efficient when samples exceed features.
    # random_state=42 makes the fit reproducible when randomness is used.
    model = LinearSVC(C=c_value, max_iter=10000, dual=False, random_state=42)
    model.fit(X_train_scaled, y_train)

    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)

    results.append(
        {
            "C": c_value,
            "train_accuracy": round(accuracy_score(y_train, train_predictions), 3),
            "test_accuracy": round(accuracy_score(y_test, test_predictions), 3),
            "coefficient_l2_norm": round(float((model.coef_ ** 2).sum() ** 0.5), 3),
        }
    )

results_table = pd.DataFrame(results)
results_table["gap"] = (results_table["train_accuracy"] - results_table["test_accuracy"]).round(3)

print("LinearSVC regularization controlled by C")
print(
    "\nWhat C does in LinearSVC:"
    "\n- small C applies stronger regularization"
    "\n- large C allows a tighter fit to the training data"
    "\n- this changes the size of the learned weight vector"
)

print(
    "\nWhy this comparison matters:"
    "\n- it shows regularization without changing the core model family"
    "\n- train/test gaps help spot underfitting and overfitting"
    "\n- coefficient norms help show how aggressive the penalty is"
)

print("\nResults across C values:")
print(results_table.to_string(index=False))

print(
    "\nInterpretation: in linear SVMs, C plays the same strategic role as many"
    " regularization hyperparameters elsewhere: it controls how hard the model"
    " pushes to fit the training data."
)
