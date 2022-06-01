import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('french_coal_production.xlsx')
print(df.dtypes)

df['converted_units'] = df["num_with_units"].replace({"K":"*1e3", "M":"*1e6"}, regex=True).map(pd.eval).astype(int)
df['converted_units'] = df['converted_units'].div(1000).round(1)
print(df.head(50))

sns.scatterplot(x='date', y='converted_units', data=df, hue='type')
plt.show()



def min_max_normalize(lst):
  minimum = min(lst)
  maximum = max(lst)
  normalized = []
  for element in lst:
    normalized_num = (element - minimum) / (maximum - minimum)
    normalized.append(normalized_num)
  return normalized

#print(min_max_normalize(release_dates))