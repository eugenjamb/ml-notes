import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# ============================================================
# Gini Impurity And Split Selection
# ============================================================
#
# What this method does:
# - measures how mixed a node is
# - compares candidate splits by how much they reduce impurity
# - helps explain how decision trees choose questions
#
# Why we use it:
# - it shows the logic behind tree construction
# - it connects manual calculations to sklearn behavior
# - it makes "best split" more interpretable than treating trees like magic


def gini(data):
    """Calculate the Gini impurity for a target column."""
    data = pd.Series(data)
    return 1 - sum(data.value_counts(normalize=True) ** 2)


def info_gain(left, right, current_impurity):
    """Calculate the impurity reduction from a candidate split."""
    weight_left = float(len(left)) / (len(left) + len(right))
    return current_impurity - weight_left * gini(left) - (1 - weight_left) * gini(right)


def build_custom_dataframe():
    """Create a small categorical dataset directly in the file."""
    return pd.DataFrame(
        [
            ["sunny", "hot", "high", "weak", 0],
            ["sunny", "hot", "high", "strong", 0],
            ["overcast", "hot", "high", "weak", 1],
            ["rain", "mild", "high", "weak", 1],
            ["rain", "cool", "normal", "weak", 1],
            ["rain", "cool", "normal", "strong", 0],
            ["overcast", "cool", "normal", "strong", 1],
            ["sunny", "mild", "high", "weak", 0],
            ["sunny", "cool", "normal", "weak", 1],
            ["rain", "mild", "normal", "weak", 1],
            ["sunny", "mild", "normal", "strong", 1],
            ["overcast", "mild", "high", "strong", 1],
            ["overcast", "hot", "normal", "weak", 1],
            ["rain", "mild", "high", "strong", 0],
        ],
        columns=["outlook", "temperature", "humidity", "wind", "play"],
    )


def main():
    df = build_custom_dataframe()
    print("Custom dataset:")
    print(df)

    X = pd.get_dummies(df.iloc[:, 0:4])
    y = df["play"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.2
    )

    root_gini = gini(y_train)
    print(f"\nGini impurity at root: {root_gini:.4f}")

    info_gain_list = []
    for column in x_train.columns:
        left = y_train[x_train[column] == 0]
        right = y_train[x_train[column] == 1]
        info_gain_list.append([column, info_gain(left, right, root_gini)])

    info_gain_table = (
        pd.DataFrame(info_gain_list, columns=["feature", "impurity_gain"])
        .sort_values("impurity_gain", ascending=False)
        .reset_index(drop=True)
    )
    print("\nImpurity gain for each possible split:")
    print(info_gain_table)
    print("\nBest split based on Gini impurity:")
    print(info_gain_table.iloc[0])

    model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0)
    model.fit(x_train, y_train)

    print(f"\nModel accuracy on test set: {model.score(x_test, y_test):.4f}")
    print("\nDecision tree learned by sklearn:")
    print(export_text(model, feature_names=list(X.columns)))


if __name__ == "__main__":
    main()
