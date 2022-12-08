import numpy as np
import pandas as pd
import pylab as plt

data = pd.read_excel("sheet1与sheet2合成分析.xlsx",sheet_name = '0与1',header = 0,index_col = 0)
data = data[['二氧化硅(SiO2)','氧化钾(K2O)','氧化钙(CaO)','氧化铁(Fe2O3)','氧化铅(PbO)','氧化钡(BaO)','五氧化二磷(P2O5)','氧化锶(SrO)']]
D = data
flag = ['二氧化硅(SiO2)','氧化钾(K2O)','氧化钙(CaO)','氧化铁(Fe2O3)','氧化铅(PbO)','氧化钡(BaO)','五氧化二磷(P2O5)','氧化锶(SrO)']
data = data.values
p1 = data[:]
p11 = p1[:8,:]
p12 = p1[8:14,:]
p13 = p1[14:25,:]
p14 = p1[25:,:]

fig = []

for i in range(8):
    plt.figure(i)
    plt.subplot(221)
    plt.plot(p11[:,i],marker = 'o',color = 'tomato')
    plt.subplot(222)
    plt.plot(p12[:,i],marker = 'o',color = 'lime')
    plt.subplot(223)
    plt.plot(p13[:,i],marker = 'o',color = 'deepskyblue')
    plt.subplot(224)
    plt.plot(p14[:,i],marker = 'o',color = 'gold')
    plt.savefig(str(i))


    print('#',flag[i])
    print('最大值',np.max(p11[:,i]),np.max(p12[:,i]),np.max(p13[:,i]),np.max(p14[:,i]))
    print('最小值',np.min(p11[:,i]),np.min(p12[:,i]),np.min(p13[:,i]),np.min(p14[:,i]))
    print('均值',np.mean(p11[:,i]),np.mean(p12[:,i]),np.mean(p13[:,i]),np.mean(p14[:,i]))
    print('极差',np.ptp(p11[:,i]),np.ptp(p12[:,i]),np.ptp(p13[:,i]),np.ptp(p14[:,i]))


    fig.append(  [np.max(p11[:,i]),np.max(p12[:,i]),np.max(p13[:,i]),np.max(p14[:,i])])
    fig.append( [np.min(p11[:,i]),np.min(p12[:,i]),np.min(p13[:,i]),np.min(p14[:,i])])
    fig.append( [np.mean(p11[:,i]),np.mean(p12[:,i]),np.mean(p13[:,i]),np.mean(p14[:,i])])
    fig.append([np.ptp(p11[:,i]),np.ptp(p12[:,i]),np.ptp(p13[:,i]),np.ptp(p14[:,i])])

fig = np.array(fig)

fig = pd.DataFrame(fig)
fig.to_excel('统计图.xlsx')



