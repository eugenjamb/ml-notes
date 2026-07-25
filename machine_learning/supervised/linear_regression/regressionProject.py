# Import the required libraries
import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# ============================================================
# Linear Regression Trend Projection
# ============================================================
#
# What this method does:
# - fits a straight line to average honey production by year
# - summarizes the long-term trend with slope and intercept
# - extends that line forward to make simple future projections
#
# Why we use it:
# - it is an easy first model for trend data
# - it shows how regression can be used for forecasting-style questions
# - it highlights the difference between fitting history and extrapolating beyond it

# Create a linear regression model
regr = linear_model.LinearRegression()

# Load the honey production dataset
df = pd.read_csv(
    "https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv"
)

# Display the first 5 rows of the dataset
print(df.head())

# Group the data by year and calculate the average honey production for each year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# Select the year column as the independent variable (X)
X = prod_per_year.year

# Convert X into a 2D array because scikit-learn requires this format
X = X.values.reshape(-1, 1)

# Select total honey production as the dependent variable (y)
y = prod_per_year.totalprod

# Train (fit) the linear regression model using X and y
regr.fit(X, y)

# Print the slope (coefficient) of the regression line
print(regr.coef_)

# Print the y-intercept of the regression line
print(regr.intercept_)

# Use the model to predict honey production for existing years
y_predict = regr.predict(X)

# Create an array of future years from 2013 to 2050
X_future = np.array(range(2013, 2051))

# Reshape into a 2D array for scikit-learn
X_future = X_future.reshape(-1, 1)

# Predict honey production for future years
future_predict = regr.predict(X_future)

# Create a scatter plot of the original data
plt.scatter(X, y)

# Plot the regression line for the original data
plt.plot(X, y_predict)

# Plot the predicted future honey production trend
plt.plot(X_future, future_predict)

# Display the graph
plt.show()
