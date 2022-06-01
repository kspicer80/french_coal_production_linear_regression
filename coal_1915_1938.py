import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_excel('french_coal_production_1915_1938.xlsx')
print(df.head())
print(df.tail())
print(df.dtypes)

X = df['date']
X = X.values.reshape(-1, 1)

y = df['metric_tons']
print(y[0:5])

plt.plot(X, y, color='cyan', marker='o')
plt.title("French Coal Production 1915-1939")
plt.xlabel("Year")
plt.ylabel("Coal Production in Metric Tons")

regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.coef_, regr.intercept_)

y_predict = regr.predict(X)
plt.plot(X, y_predict, color='red')
plt.title("French Coal Production Linear Regression")
plt.xlabel("Year")
plt.ylabel("Coal Production in Millions of Metric Tons")

X_future = np.array(range(1918, 2022))
X_future = X_future.reshape(-1, 1)

future_predict = regr.predict(X_future)
plt.plot(X_future, future_predict, color='green')
plt.title("French Coal Production Linear Regression Predicted from 1915-1938 Data")
plt.ylabel("Coal Production in Millions of Metric Tons")
plt.xlabel("Year")
plt.show()