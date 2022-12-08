import pandas as pd
import pylab as plt
import numpy as np

data = pd.read_excel('第一问第三小问铅钡.xlsx',header = 0)

data = data.values[:,3:]

data = data[:,:11]

for i in range(11):
    plt.figure(i)
    plt.scatter(range(31),data[:,i])
    print("#",str(i))
    print('平均值为',np.mean(data[:11,i]))
    print('中位数为',np.median(data[:11,i]))