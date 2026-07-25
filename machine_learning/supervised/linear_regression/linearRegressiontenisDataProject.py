import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ------------------------
# Linear Regression On Tennis Statistics
# ------------------------
#
# What this method does:
# - uses player stats to predict winnings
# - compares single-feature, two-feature, and multi-feature models
# - measures how well each feature set explains the target
#
# Why we use it:
# - it shows that regression quality depends heavily on feature choice
# - it connects visual exploration with model building
# - it introduces train/test evaluation for multiple candidate models


# ------------------------
# LOAD DATA
# ------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "tennis_stats.csv"

statistics = pd.read_csv(DATA_PATH)
print(statistics.head())

print(statistics.shape)

print(statistics.columns)

print(statistics.describe())


# ------------------------
# EXPLORATORY ANALYSIS
# ------------------------

# Wins vs Winnings

plt.scatter(
    statistics["Wins"],
    statistics["Winnings"]
)

plt.xlabel("Wins")

plt.ylabel("Winnings")

plt.title("Wins vs Winnings")

plt.show()


# BreakPointsOpportunities vs Winnings

plt.scatter(
    statistics["BreakPointsOpportunities"],
    statistics["Winnings"]
)

plt.xlabel("BreakPointsOpportunities")

plt.ylabel("Winnings")

plt.title("BreakPointsOpportunities vs Winnings")

plt.show()


# TotalPointsWon vs Winnings

plt.scatter(
    statistics["TotalPointsWon"],
    statistics["Winnings"]
)

plt.xlabel("TotalPointsWon")

plt.ylabel("Winnings")

plt.title("TotalPointsWon vs Winnings")

plt.show()


# Ranking vs Winnings

plt.scatter(
    statistics["Ranking"],
    statistics["Winnings"]
)

plt.xlabel("Ranking")

plt.ylabel("Winnings")

plt.title("Ranking vs Winnings")

plt.show()


# ------------------------
# CORRELATION
# ------------------------

numeric_statistics = statistics.select_dtypes(
    include=["number"]
)

corr = numeric_statistics.corr()

print(
    corr["Winnings"]
    .sort_values(ascending=False)
)

plt.figure(figsize=(12,8))

sns.heatmap(
    corr,
    cmap="coolwarm"
)

plt.show()


# ------------------------
# FUNCTION TO BUILD MODELS
# ------------------------

def build_model(features, outcome):

    X = statistics[features]

    y = statistics[outcome]

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(
        x_train,
        y_train
    )

    score = model.score(
        x_test,
        y_test
    )

    print("--------------------------------")

    print("Features:", features)

    print("Outcome:", outcome)

    print("R²:", score)

    return score


# ------------------------
# SINGLE FEATURE MODELS
# ------------------------

build_model(
    ["Wins"],
    "Winnings"
)

build_model(
    ["BreakPointsOpportunities"],
    "Winnings"
)

build_model(
    ["ServiceGamesWon"],
    "Winnings"
)

build_model(
    ["TotalPointsWon"],
    "Winnings"
)


# ------------------------
# TWO FEATURE MODELS
# ------------------------

build_model(
    ["Wins", "BreakPointsOpportunities"],
    "Winnings"
)

build_model(
    ["Wins", "ServiceGamesWon"],
    "Winnings"
)

build_model(
    ["Wins", "TotalPointsWon"],
    "Winnings"
)

build_model(
    ["ServiceGamesWon", "TotalPointsWon"],
    "Winnings"
)


# ------------------------
# MULTIPLE FEATURE MODELS
# ------------------------

build_model(
[
"Wins",

"BreakPointsOpportunities",

"ServiceGamesWon",

"TotalPointsWon",

"Aces"

],
"Winnings"
)

build_model(
[
"Wins",

"BreakPointsOpportunities",

"ServiceGamesWon",

"ReturnGamesWon",

"TotalPointsWon",

"Aces"

],
"Winnings"
)
