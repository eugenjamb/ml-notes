# Python Multiple Linear Regression (StreetEasy) - Study Notes
#
# What this method does:
# - predicts one numeric target using several input features at once
# - learns one coefficient per feature plus an intercept
# - combines those values into a single regression equation
#
# Why we use it:
# - many real problems depend on several variables, not just one
# - it helps estimate the separate contribution of each feature
# - it is a strong baseline before moving to more flexible models

# =====================================
# 1. IMPORT LIBRARIES
# =====================================


# Used for plotting graphs
import matplotlib.pyplot as plt

# Used for mathematical arrays
import numpy as np

# Used to work with tables (DataFrames)
import pandas as pd

# Machine Learning functions
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Used to calculate prediction error
from sklearn.metrics import mean_squared_error


# =====================================
# 2. LOAD THE DATASET
# =====================================

# Download the Manhattan apartment dataset

streeteasy = pd.read_csv(
    "https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv"
)

# Convert to DataFrame

df = pd.DataFrame(streeteasy)


# =====================================
# 3. SELECT FEATURES (X)
# =====================================

# Features are the INPUT variables used to predict rent

x = df[[
    'bedrooms',
    'bathrooms',
    'size_sqft',
    'min_to_subway',
    'floor',
    'building_age_yrs',
    'no_fee',
    'has_roofdeck',
    'has_washer_dryer',
    'has_doorman',
    'has_elevator',
    'has_dishwasher',
    'has_patio',
    'has_gym'
]]

# TARGET VARIABLE

# This is what we want to predict

y = df[['rent']]


# =====================================
# 4. SPLIT THE DATA
# =====================================

# 80% -> training data
# 20% -> testing data

# random_state makes the split reproducible

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=6
)


# =====================================
# 5. CREATE THE MODEL
# =====================================

# Create a Linear Regression model

mlr = LinearRegression()


# =====================================
# 6. TRAIN THE MODEL
# =====================================

# Learn from the 80% training data

mlr.fit(x_train, y_train)


# =====================================
# 7. MAKE PREDICTIONS
# =====================================

# Predict rent using unseen data (20%)

y_predicted = mlr.predict(x_test)

print(y_predicted)


# =====================================
# 8. INTERCEPT
# =====================================

# Starting value of the equation

print(mlr.intercept_)


# =====================================
# 9. SLOPES (COEFFICIENTS)
# =====================================

# Shows how each feature affects rent

print(mlr.coef_)


# Print feature names with coefficients

for feature, coef in zip(x.columns, mlr.coef_[0]):
    print(feature, ":", coef)


# =====================================
# 10. PRINT THE EQUATION
# =====================================

print("Rent =", mlr.intercept_[0])

for feature, coef in zip(x.columns, mlr.coef_[0]):
    print(f"+ ({coef}) * {feature}")


# Conceptually:

# rent =
# intercept
# + bedrooms * coefficient
# + bathrooms * coefficient
# + size_sqft * coefficient
# + ...


# =====================================
# 11. TRAIN AND TEST SCORES
# =====================================

# R² score (coefficient of determination)

print("Train score:")
print(mlr.score(x_train, y_train))

print("Test score:")
print(mlr.score(x_test, y_test))


# Interpretation:

# 1.0 = perfect

# 0.9 = excellent

# 0.8 = very good

# 0.7 = good

# 0.5 = moderate

# 0 = no predictive power

# <0 = worse than guessing


# =====================================
# 12. MEAN SQUARED ERROR (OPTIONAL)
# =====================================

mse = mean_squared_error(y_test, y_predicted)

print("MSE:", mse)


# =====================================
# 13. SCATTER PLOT
# =====================================

# Actual rents vs predicted rents

plt.scatter(
    y_test,
    y_predicted,
    alpha=0.4
)

# Perfect prediction line

plt.plot(
    range(20000),
    range(20000)
)

plt.xlabel("Actual Rent ($Y_i$)")

plt.ylabel("Predicted Rent ($Ŷ_i$)")

plt.title("Actual Rent vs Predicted Rent")

plt.show()


# =====================================
# 14. PREDICT A NEW APARTMENT
# =====================================

zoe_apartment = [[

    1,      # bedrooms

    1,      # bathrooms

    620,    # size_sqft

    16,     # min_to_subway

    1,      # floor

    98,     # building_age_yrs

    0,      # no_fee

    0,      # has_roofdeck

    1,      # has_washer_dryer

    0,      # has_doorman

    0,      # has_elevator

    0,      # has_dishwasher

    1,      # has_patio

    0       # has_gym

]]

prediction = mlr.predict(zoe_apartment)

print("Predicted rent: $%.2f" % prediction)


# =====================================
# 15. FIND CORRELATIONS
# =====================================

# Helps decide which features are useful

print(
    df.corr()['rent']
      .sort_values(ascending=False)
)


# Features near:

# +1 -> strong positive relationship

# -1 -> strong negative relationship

#  0 -> weak relationship


# =====================================
# QUICK MEMORY SHEET
# =====================================

# train_test_split()
# -> split data

# random_state
# -> same shuffle every run

# fit()
# -> train the model

# predict()
# -> make predictions

# intercept_
# -> starting value

# coef_
# -> slopes

# score()
# -> R² accuracy

# mean_squared_error()
# -> prediction error

# scatter()
# -> draw dots

# alpha
# -> transparency

# corr()
# -> feature relationships

