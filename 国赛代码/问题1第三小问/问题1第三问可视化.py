import pandas as pd
import pylab as plt

data = pd.read_excel('sheet1与sheet2合成分析.xlsx',sheet_name = '高钾',header = 0)
data = data.values
x = data[:,3:-1]
plt.rc('font',family = 'SimHei')
for i in range(11):
    plt.figure(i)
    plt.scatter(range(12),x[:12,i],label = '高钾风化前')
    plt.scatter(range(12,18),x[12:,i],label = '高钾风化后')
    plt.legend()
    plt.savefig(str(i))
    plt.show()