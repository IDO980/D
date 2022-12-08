import pylab as plt
import numpy as np

data1 = []
for i in range(6):
     data1.append(np.random.randint(6,10))

data2 = []
for i in range(4):
     data2.append(np.random.randint(1, 4))

plt.figure(1)
plt.scatter(range(6), data1)
plt.scatter(range(4), data2)
plt.savefig('距离型数据')

data3 = []
for i in range(6):
    data3.append(np.random.randint(4,8))
data4 = []
for i in range(6):
    data4.append(np.random.randint(2,7))

plt.figure(2)
plt.scatter(range(6),data3)
plt.scatter(range(6),data4)
plt.savefig('交叉型数据')

plt.show()

