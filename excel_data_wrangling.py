import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def min_max_normalize(lst):
  minimum = min(lst)
  maximum = max(lst)
  normalized = []
  for element in lst:
    normalized_num = (element - minimum) / (maximum - minimum)
    normalized.append(normalized_num)
  return normalized

def to_million(x):
    return abs(x/1000000)
    
df = pd.read_excel('french_coal_production.xlsx')
print(df.dtypes)
print(df.tail(25))

df['converted_units'] = df["num_with_units"].replace({"K":"*1e3", "M":"*1e6"}, regex=True).map(pd.eval).astype(int)
df['converted_units'] = df['converted_units'].apply(to_million)
print(df.head(50))

df_min_max_scaled = df.copy()
#df_min_max_scaled['converted units'] = (df_min_max_scaled['converted_units'] - df_min_max_scaled['converted_units'].min()) / (df_min_max_scaled['converted_units'].max() - df_min_max_scaled['converted_units'].min())
df_sklearn = df.copy()
df_sklearn['converted_units'] = MinMaxScaler().fit_transform(np.array(df_sklearn['converted_units']).reshape(-1, 1))
#df = df['converted_units'].apply(lambda x: min_max_normalize(x))
sns.scatterplot(x='date', y='converted_units', data=df_sklearn, hue='type')
plt.show()


#print(min_max_normalize(release_dates))