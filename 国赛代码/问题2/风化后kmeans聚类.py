import pandas as pd
from sklearn.cluster import KMeans
data = pd.read_excel('风化后.xlsx',header = 0).values
data = data[:,3:]
md = KMeans(2).fit(data)
print(md.labels_)