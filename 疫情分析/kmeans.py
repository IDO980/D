import numpy as np
import pandas as pd
import pylab as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

australia = pd.read_excel('1.xlsx',sheet_name='australia',header= 0,index_col=0)
american = pd.read_excel('1.xlsx',sheet_name='american',header= 0,index_col=0)
china = pd.read_excel('1.xlsx',sheet_name='china',header= 0,index_col=0)
english = pd.read_excel('1.xlsx',sheet_name='english',header= 0,index_col=0)
russia = pd.read_excel('1.xlsx',sheet_name='russia',header= 0,index_col=0)
switzerland = pd.read_excel('1.xlsx',sheet_name='switzerland',header= 0,index_col=0)

ch =   china.new_cases_smoothed_per_million.fillna(0)  #中国值较小，因此缺失值直接补0
en =   english.new_cases_smoothed_per_million.fillna(method='backfill')  #英国值很大，因此缺失值补插值
au =  australia.new_cases_smoothed_per_million.fillna(method='backfill')  #中国值较小，因此缺失值直接补0
am =  american.new_cases_smoothed_per_million.fillna(method='backfill')  #中国值较小，因此缺失值直接补0
ru =  russia.new_cases_smoothed_per_million.fillna(method='backfill')  #中国值较小，因此缺失值直接补0
sw = switzerland.fillna(value = {'new_cases_smoothed_per_million':switzerland.new_cases_smoothed_per_million.interpolate()})
sw = sw.new_cases_smoothed_per_million


am2 = american.new_deaths_smoothed_per_million.fillna(method='backfill')
en2 = english.new_deaths_smoothed_per_million.fillna(method='backfill')
au2 = australia.new_deaths_smoothed_per_million.fillna(method='backfill')
ch2 = china.new_deaths_smoothed_per_million.fillna(method='backfill')
ru2 = russia.new_deaths_smoothed_per_million.fillna(method='backfill')
sw2 = switzerland.new_deaths_smoothed_per_million.fillna(method='backfill')

ch3 = china.fillna(value = {'people_fully_vaccinated':china.people_fully_vaccinated.interpolate()})
en3 = english.fillna(value = {'people_fully_vaccinated':english.people_fully_vaccinated.interpolate()})
au3 = australia.fillna(value = {'people_fully_vaccinated':australia.people_fully_vaccinated.interpolate()})
am3 = american.fillna(value = {'people_fully_vaccinated':american.people_fully_vaccinated.interpolate()})
ru3 = russia.fillna(value = {'people_fully_vaccinated':russia.people_fully_vaccinated.interpolate()})
sw3 = switzerland.fillna(value = {'people_fully_vaccinated':switzerland.people_fully_vaccinated.interpolate()})
ch3 = ch3.people_fully_vaccinated/ch3.population
en3 = en3.people_fully_vaccinated/en3.population
am3 = am3.people_fully_vaccinated/am3.population
au3 = au3.people_fully_vaccinated/au3.population
ru3 = ru3.people_fully_vaccinated/ru3.population
sw3 = sw3.people_fully_vaccinated/sw3.population




am = np.matrix(am).T
am2 = np.matrix(am2).T
am3 = np.matrix(am3).T

au = np.matrix(au).T
au2 = np.matrix(au2).T
au3= np.matrix(au3).T

ch= np.matrix(ch).T
ch2= np.matrix(ch2).T
ch3= np.matrix(ch3).T

en = np.matrix(en).T
en2 = np.matrix(en2).T
en3 = np.matrix(en3).T

ru = np.matrix(ru).T
ru2 = np.matrix(ru2).T
ru3 = np.matrix(ru3).T

sw = np.matrix(sw).T
sw2 = np.matrix(sw2).T
sw3 = np.matrix(sw3).T


Am = np.hstack([am,am2,am3])
Au = np.hstack([au,au2,au3])
Ch = np.hstack([ch,ch2,ch3])
En = np.hstack([en,en2,en3])
Ru = np.hstack([ru,ru2,ru3])
Sw = np.hstack([sw,sw2,sw3])
Au = pd.DataFrame(Au)
Au = Au.fillna(method= 'ffill')
Ru = pd.DataFrame(Ru)
Ru = Ru.fillna(method= 'ffill')
En = pd.DataFrame(En)
En = En.fillna(method= 'ffill')
Sw = pd.DataFrame(Sw)
Sw = Sw.fillna(method= 'ffill')




Am_mean = Am.mean(axis = 0)
AU_mean = Au.mean(axis = 0)
Ch_mean = Ch.mean(axis = 0)
En_mean = En.mean(axis = 0)
Ru_mean = Ru.mean(axis = 0)
Sw_mean = Sw.mean(axis = 0)
#确定最佳聚类数
a = np.vstack([Am_mean,AU_mean,Ch_mean,Ru_mean,En_mean,Sw_mean])
b=(a-a.min(axis=0))/(a.max(axis=0)-a.min(axis=0))    #数据规格化处理
S = []; K = range(2, len(a))
for i in K:
    md = KMeans(i).fit(b)
    labels = md.labels_
    S.append(silhouette_score(b, labels))
plt.figure(1)
plt.plot(K, S,'o-')
plt.savefig('最优聚类数')
md = KMeans(4).fit(b)
labels = md.labels_
centers = md.cluster_centers_
print(centers)
plt.figure(2)
ax = plt.subplot(111,projection='3d')
ax.scatter(centers[:,0],centers[:,1],centers[:,2],color  = 'blue')
plt.savefig('聚类结果1')
plt.figure(3)
ax = plt.subplot(111,projection='3d')
ax.scatter(b[:,0],b[:,1],b[:,2],color = 'red')
plt.savefig('聚类结果2')
plt.figure(4)
ax = plt.subplot(111,projection='3d')
ax.scatter(centers[:,0],centers[:,1],centers[:,2],color  = 'blue')
ax.scatter(b[:,0],b[:,1],b[:,2],color = 'red')
plt.savefig('聚类结果3')
plt.show()