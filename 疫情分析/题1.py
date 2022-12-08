import numpy as np
import pandas as pd
import pylab as plt

australia = pd.read_excel('1.xlsx',sheet_name='australia',header= 0,index_col=0)
american = pd.read_excel('1.xlsx',sheet_name='american',header= 0,index_col=0)
china = pd.read_excel('1.xlsx',sheet_name='china',header= 0,index_col=0)
english = pd.read_excel('1.xlsx',sheet_name='english',header= 0,index_col=0)
russia = pd.read_excel('1.xlsx',sheet_name='russia',header= 0,index_col=0)
switzerland = pd.read_excel('1.xlsx',sheet_name='switzerland',header= 0,index_col=0)
#每百万人新增感染人数
ch =   china.new_cases_smoothed_per_million.fillna(0)  #中国值较小，因此缺失值直接补0
en =   english.new_cases_smoothed_per_million.fillna(method='backfill')  #英国值很大，因此缺失值补插值
au =  australia.new_cases_smoothed_per_million.fillna(method='backfill')  #中国值较小，因此缺失值直接补0
am =  american.new_cases_smoothed_per_million.fillna(method='backfill')  #中国值较小，因此缺失值直接补0
ru =  russia.new_cases_smoothed_per_million.fillna(method='backfill')  #中国值较小，因此缺失值直接补0
sw = switzerland.fillna(value = {'new_cases_smoothed_per_million':switzerland.new_cases_smoothed_per_million.interpolate()})
sw = sw.new_cases_smoothed_per_million
plt.figure(1)
plt.plot(range(len(ch)),ch,linewidth = 2,label = 'china',color = 'tomato')
plt.plot(range(len(en)),en,linewidth = 2,label = 'english',color = 'deepskyblue')
plt.plot(range(len(au)),au,linewidth = 2,label = 'australia',color = 'turquoise')
plt.plot(range(len(am)),am,linewidth = 2,label = 'american',color = 'springgreen')
plt.plot(range(len(ru)),ru,linewidth = 2,label = 'russia',color = 'moccasin')
plt.plot(range(len(sw)),sw,linewidth = 2,label = 'switzerland',color = 'lightcoral')
plt.legend()
plt.savefig('每百万人新增感染人数')
#每百万人死亡人数
am2 = american.new_deaths_smoothed_per_million.fillna(method='backfill')
en2 = english.new_deaths_smoothed_per_million.fillna(method='backfill')
au2 = australia.new_deaths_smoothed_per_million.fillna(method='backfill')
ch2 = china.new_deaths_smoothed_per_million.fillna(method='backfill')
ru2 = russia.new_deaths_smoothed_per_million.fillna(method='backfill')
sw2 = switzerland.new_deaths_smoothed_per_million.fillna(method='backfill')



plt.figure(2)
plt.plot(range(len(ch2)),ch2,linewidth = 2,label = 'china',color = 'tomato')
plt.plot(range(len(en2)),en2,linewidth = 2,label = 'english',color = 'deepskyblue')
plt.plot(range(len(au2)),au2,linewidth = 2,label = 'australia',color = 'turquoise')
plt.plot(range(len(am2)),am2,linewidth = 2,label = 'american',color = 'springgreen')
plt.plot(range(len(ru2)),ru2,linewidth = 2,label = 'russia',color = 'moccasin')
plt.plot(range(len(sw2)),sw2,linewidth = 2,label = 'switzerland',color = 'lightcoral')
plt.legend()
plt.savefig('每百万人新增死亡人数')
#每百人新增疫苗接种
ch3 = china.fillna(value = {'people_fully_vaccinated':china.people_fully_vaccinated.interpolate()})
en3 = english.fillna(value = {'people_fully_vaccinated':english.people_fully_vaccinated.interpolate()})
au3 = australia.fillna(value = {'people_fully_vaccinated':australia.people_fully_vaccinated.interpolate()})
am3 = american.fillna(value = {'people_fully_vaccinated':american.people_fully_vaccinated.interpolate()})
ru3 = russia.fillna(value = {'people_fully_vaccinated':russia.people_fully_vaccinated.interpolate()})
sw3 = switzerland.fillna(value = {'people_fully_vaccinated':switzerland.people_fully_vaccinated.interpolate()})
ch3.people_fully_vaccinated = ch3.people_fully_vaccinated/ch3.population
en3.people_fully_vaccinated = en3.people_fully_vaccinated/en3.population
am3.people_fully_vaccinated = am3.people_fully_vaccinated/am3.population
au3.people_fully_vaccinated = au3.people_fully_vaccinated/au3.population
ru3.people_fully_vaccinated = ru3.people_fully_vaccinated/ru3.population
sw3.people_fully_vaccinated = sw3.people_fully_vaccinated/sw3.population
plt.figure(3)
plt.plot(range(len(ch3.people_fully_vaccinated)),ch3.people_fully_vaccinated,linewidth = 2,label = 'china',color = 'tomato')
plt.plot(range(len(en3.people_fully_vaccinated)),en3.people_fully_vaccinated,linewidth = 2,label = 'english',color = 'tomato')
plt.plot(range(len(au3.people_fully_vaccinated)),au3.people_fully_vaccinated,linewidth = 2,label = 'australia',color = 'deepskyblue')
plt.plot(range(len(am3.people_fully_vaccinated)),am3.people_fully_vaccinated,linewidth = 2,label = 'american',color = 'turquoise')
plt.plot(range(len(ru3.people_fully_vaccinated)),ru3.people_fully_vaccinated,linewidth = 2,label = 'russia',color = 'springgreen')
plt.plot(range(len(sw3.people_fully_vaccinated)),sw3.people_fully_vaccinated,linewidth = 2,label = 'switzerland',color = 'moccasin')
plt.legend()
plt.savefig('每百人新增疫苗接种')


plt.figure(4)
plt.subplot(311)
plt.plot(range(len(ch)),ch,linewidth = 2,label = 'china',color = 'tomato')
plt.plot(range(len(en)),en,linewidth = 2,label = 'english',color = 'deepskyblue')
plt.plot(range(len(au)),au,linewidth = 2,label = 'australia',color = 'turquoise')
plt.plot(range(len(am)),am,linewidth = 2,label = 'american',color = 'springgreen')
plt.plot(range(len(ru)),ru,linewidth = 2,label = 'russia',color = 'moccasin')
plt.plot(range(len(sw)),sw,linewidth = 2,label = 'switzerland',color = 'lightcoral')

plt.subplot(312)
plt.plot(range(len(ch2)),ch2,linewidth = 2,label = 'china',color = 'tomato')
plt.plot(range(len(en2)),en2,linewidth = 2,label = 'english',color = 'deepskyblue')
plt.plot(range(len(au2)),au2,linewidth = 2,label = 'australia',color = 'turquoise')
plt.plot(range(len(am2)),am2,linewidth = 2,label = 'american',color = 'springgreen')
plt.plot(range(len(ru2)),ru2,linewidth = 2,label = 'russia',color = 'moccasin')
plt.plot(range(len(sw2)),sw2,linewidth = 2,label = 'switzerland',color = 'lightcoral')

plt.subplot(313)
plt.plot(range(len(ch3.people_fully_vaccinated)),ch3.people_fully_vaccinated,linewidth = 2,label = 'china',color = 'tomato')
plt.plot(range(len(en3.people_fully_vaccinated)),en3.people_fully_vaccinated,linewidth = 2,label = 'english',color = 'deepskyblue')
plt.plot(range(len(au3.people_fully_vaccinated)),au3.people_fully_vaccinated,linewidth = 2,label = 'australia',color = 'turquoise')
plt.plot(range(len(am3.people_fully_vaccinated)),am3.people_fully_vaccinated,linewidth = 2,label = 'american',color = 'springgreen')
plt.plot(range(len(ru3.people_fully_vaccinated)),ru3.people_fully_vaccinated,linewidth = 2,label = 'russia',color = 'moccasin')
plt.plot(range(len(sw3.people_fully_vaccinated)),sw3.people_fully_vaccinated,linewidth = 2,label = 'switzerland',color = 'lightcoral')
plt.savefig('对比图')





plt.show()





