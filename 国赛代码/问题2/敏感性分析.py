import pandas as pd
from sklearn.cluster import KMeans

data1 = pd.read_excel('风化后.xlsx',header = 0).values
data1 = data1[:,3:].copy()
per_data1 = data1.copy()

data2 = pd.read_excel('风化前.xlsx',header = 0).values
data2 = data2[:,3:].copy()
per_data2 = data2.copy()


data1 = per_data1.copy()
data1[:,0] = data1[:,0].copy() * 1.01

md = KMeans(2).fit(data1)
labels = md.labels_
print(labels)

data1 = per_data1.copy()
data1[:,0] = data1[:,0].copy() * 1.05

md = KMeans(2).fit(data1)
labels = md.labels_
print(labels)

data1 = per_data1.copy()
data1[:,0] = data1[:,0].copy() * 1.1

md = KMeans(2).fit(data1)
labels = md.labels_
print(labels)


data1 = per_data1.copy()
data1[:,1] = data1[:,1].copy() * 1.05

md = KMeans(2).fit(data1)
labels = md.labels_
print(labels)

data1 = per_data1.copy()
data1[:,1] = data1[:,1].copy() * 1.01

md = KMeans(2).fit(data1)
labels = md.labels_
print(labels)

data1 = per_data1.copy()
data1[:,1] = data1[:,1].copy() * 1.11

md = KMeans(2).fit(data1)
labels = md.labels_
print('有误判',labels)

data2 = per_data2.copy()
data2[:,-1] = data2[:,-1].copy() * 1.01

md = KMeans(2).fit(data2)
labels = md.labels_

print(labels)

data2 = per_data2.copy()
data2[:,-1] = data2[:,-1].copy() * 1.05

md = KMeans(2).fit(data2)
labels = md.labels_
print(labels)

data2 = per_data2.copy()
data2[:,-1] = data2[:,-1].copy() * 1.1

md = KMeans(2).fit(data2)
labels = md.labels_
print(labels)

data2 = per_data2.copy()
data2[:,-2] = data2[:,-2].copy() * 1.01

md = KMeans(2).fit(data2)
labels = md.labels_
print(labels)


data2 = per_data2.copy()
data2[:,-2] = data2[:,-2].copy() * 1.05

md = KMeans(2).fit(data2)
labels = md.labels_
print(labels)

data2 = per_data2.copy()
data2[:,-2] = data2[:,-2].copy() * 1.1

md = KMeans(2).fit(data2)
labels = md.labels_
print(labels)
