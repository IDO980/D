import pandas as pd
import pylab as plt
import numpy as np
data = pd.read_excel('sheet1与sheet2合成分析.xlsx',sheet_name = '高钾',header = 0)
data = data.values
x = data[:,3:-1]
for i in range(11):
    plt.figure(i)
    plt.scatter(range(18),x[:,i])
    print("#",str(i))
    print('平均值为',np.mean(x[:12,i]))
    print('中位数为',np.median(x[:12,i]))
    plt.show()