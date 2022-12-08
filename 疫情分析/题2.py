from numpy import *
import numpy as np
from pandas import *
import pylab as plt

australia = read_excel('题2.xlsx',sheet_name='australia',header= 0,index_col=None)
american = read_excel('题2.xlsx',sheet_name='american',header= 0,index_col=None)
china = read_excel('题2.xlsx',sheet_name='china',header= 0,index_col=None)
english = read_excel('题2.xlsx',sheet_name='english',header= 0,index_col=None)
russia = read_excel('题2.xlsx',sheet_name='russia',header= 0,index_col=None)
switzerland = read_excel('题2.xlsx',sheet_name='switzerland',header= 0,index_col=None)

au = australia.fillna(method = 'ffill')
am = american.fillna(method = 'ffill')
ch = china.fillna(method = 'ffill')
en = english.fillna(method = 'ffill')
ru = russia.fillna(method = 'ffill')
sw = switzerland.fillna(method = 'ffill')

au = au.values
am = am.values
ch = ch.values
en = en.values
ru = ru.values
sw = sw.values

mean_au = au.mean(axis=0)
mean_am = am.mean(axis=0)
mean_ch = ch.mean(axis=0)
mean_en = en.mean(axis=0)
mean_ru = ru.mean(axis=0)
mean_sw = sw.mean(axis=0)

a = vstack([mean_au, mean_am, mean_ch, mean_en, mean_ru, mean_sw])
b = np.linalg.norm(a,axis = 0)
R = a/b
R[:,0] = 1 - a[:,0]/b[0]
R[:,1] = 1 - a[:,1]/b[1]
#熵权法求指标的权重
n=R.shape[0]
s=R.sum(axis=0)
p=R/s  #求特征比重矩阵
e=-(p*np.log(p)).sum(axis=0)/np.log(n)  #熵值
g=1-e  #差异系数
w=g/sum(g)  #权重
plt.figure(1)
plt.rc('font',family = 'SimHei')
plt.pie(w,labels = ['每百万人新增感染人数','每百万人新增死亡人数',' 严格指数','每千人核酸检测总数'],colors = ['tomato','deepskyblue','turquoise','lightcoral'],autopct='%.2f%%',explode=(0.1,0,0,0))
plt.savefig('权重图')
print(w)
#灰色关联度评价
bp = R.max(axis=0)  #各指标的最大值（即参考序列）
c = bp - R  #参考序列与每个序列的差
m2 = c.min(); m1 = c.max()  #最小差和最大差
r = 0.5  #分辨系数
xs = (m2+r*m1)/(c+r*m1)  #灰色关联系数
print(xs)
f = w@xs.T
f = np.sort(f)[::-1]
print("关联系数=",xs,"\n关联度=",f)  #关联度越大越优
plt.figure(2)
X = ['中国','英国','澳大利亚','俄罗斯','美国','瑞士']
Y1 = f
b = plt.barh(['中国','英国','澳大利亚','俄罗斯','美国','瑞士'],f,height = 0.6 )

for rect in b:
    m=rect.get_width()
    plt.text(m,rect.get_y()+rect.get_height()/2,'%.2f'%np.round(m,2),ha='left',va='center')
plt.savefig('综合评价排名图')

c = ch.copy()
b1 = np.linalg.norm(c,axis = 0)
C = c/b1
C[:,0] = 1 - c[:,0]/b1[0]
C[:,1] = 1 - c[:,1]/b1[1]
c_s = w @ C.T

a = am.copy()
b2 = np.linalg.norm(a,axis = 0)
A = a/b2
A[:,0] = 1 - a[:,0]/b2[0]
A[:,1] = 1 - a[:,1]/b2[1]
a_s = w @ A.T
plt.figure(3)
plt.plot(c_s[200:250],linewidth =2,label = 'china',color = 'tomato')
plt.plot(a_s[200:250],linewidth =2,label = 'american',color = 'deepskyblue')
plt.ylim([0.408,0.428])
plt.grid()
plt.legend(loc = 'best')
plt.savefig('中美50天评分对比图')

plt.figure(4)
s = np.vstack([c_s,a_s]).T
plt.boxplot(s)
plt.ylim([0.39,0.428])
plt.show()