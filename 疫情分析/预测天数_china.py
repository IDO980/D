import numpy as np
import pandas as pd
import pylab as plt
from sklearn.neural_network import MLPRegressor

a = pd.read_excel('神经网络预测.xlsx',header= 0,index_col=None)
a = a.fillna(method='ffill')
a = a.values



x0 = np.arange(len(a)).reshape(-1,1); y0 = a[:,0]  #提出训练样本数据
m1 = x0.max(axis=0); m2 = x0.min(axis=0)  #计算逐列最大值和最小值
bx0 = 2*(x0-m2)/(m1-m2)-1  #数据标准化
#构造并拟合模
md = MLPRegressor(solver='lbfgs',activation='relu',
     hidden_layer_sizes = 16000,max_iter= 300).fit(bx0, y0)
x = np.arange(len(a),len(a)+7).reshape(-1,1)
bx = 2*(x-m2) / (m1-m2)-1  #数据标准化
yh = md.predict(bx); print('预测值为：,',np.round(yh,4))
yh0 = md.predict(bx0); delta = abs(yh0-y0)/y0*100
print('已知数据预测的相对误差：\n', np.round(delta,4))
t = np.arange(len(a))
plt.rc('font', size=15); plt.rc('font', family='SimHei')
plt.plot(t, y0,  color = 'red',label='原始数据')
#plt.plot(t, yh0, color = 'dodgerblue',label='预测数据')
plt.legend()
plt.show()
















