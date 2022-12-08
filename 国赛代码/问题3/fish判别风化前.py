from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import numpy as np

data = pd.read_excel('风化前.xlsx',header = 0).values
a = data[:,3:]
x = [[0,6.08,0,0],[1.36,7.19,39.58,4.69],[0.79,2.89,24.28,8.31],[0.23,0.89,21.24,11.34]]
x = np.array(x)
per_x = x.copy()
y = []
for i in range(12):
    y.append(1)
for i in range(23):
    y.append(2)
md = LDA().fit(a, y)
v = md.predict(x)
print('结果：  ',v)
print('误判率为：', 1-md.score(a, y))

x = per_x.copy()
x[:,-1] = x[:,-1].copy() * 1.01
v = md.predict(x)
print('per_0.01',v)

x = per_x.copy()
x[:,-1] = x[:,-1].copy() * 1.05

print('per_0.05',v)

x = per_x.copy()
x[:,-1] = x[:,-1].copy() * 1.1
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

print('结果正确')