import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('french_coal_production_1915_1938.xlsx')
print(df.head())
print(df.tail())
print(df.dtypes)

X = df['date']
X = X.values.reshape(-1, 1)

y = df['metric_tons']
print(y[0:5])

plt.plot(X, y, color='cyan', marker='o')
plt.title("French Coal Production 1870-1919")
plt.xlabel("Year")
plt.ylabel("Coal Production in Metric Tons")
plt.show()
