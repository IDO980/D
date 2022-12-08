import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd

data = pd.read_excel('风化后.xlsx',header = 0).values
a = data[:,3:]
x = [[37.75,34.3,0],[62.29,12.23,2.16],[93.17,0,0],[90.83,0,0]]
x = np.array(x)
per_x = x
y = []
for i in range(6):
    y.append(1)
for i in range(23):
    y.append(2)
y = np.array(y)
md = LDA(solver = 'svd',priors = None ).fit(a, y)
v = md.predict(x)
print('结果为：  ', v)
print('误判率为：', 1-md.score(a, y))

x = per_x.copy()
x[:,0] = x[:,0].copy() * 1.01
v = md.predict(x)
print('per_0.01',v)

x = per_x.copy()
x[:,0] = x[:,0].copy() * 1.05

print('per_0.05',v)

x = per_x.copy()
x[:,0] = x[:,0].copy() * 1.1
v = md.predict(x)
print('per_0.1',v)


x = per_x.copy()
x[:,-2] = x[:,-2].copy() * 1.01
v = md.predict(x)
print('per_0.01',v)

x = per_x.copy()
x[:,-2] = x[:,-2].copy() * 1.05
v = md.predict(x)
print('per_0.05',v)

x = per_x.copy()
x[:,-2] = x[:,-2].copy() * 1.1
v = md.predict(x)
print('per_0.1',v)

print("产生误判")