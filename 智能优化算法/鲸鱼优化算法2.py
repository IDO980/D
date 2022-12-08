import numpy as np
import random
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''优化函数'''



def calc_f(X):
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    x4 = X[3]
    x5 = X[4]
    y1 = np.exp(-9.5139  + 0.0250 *x1 -0.0027 *x2 -0.0590 * x3 + 0.0122 * x4 - 0.7363 * x5)/(1+np.exp(-9.5139  + 0.0250 *x1 -0.0027 *x2 -0.0590 * x3 + 0.0122 * x4 - 0.7363 * x5))
    y2 = np.exp(-7.0667 +0.0148*x1 +0.0163 *x2 - 0.3151 * x3 - 0.01 * x4 +0.2049 * x5)/(1+np.exp(-7.0667 +0.0148*x1 +0.0163 *x2 - 0.3151 * x3 - 0.01 * x4 +0.2049 * x5))
    return y1 * y2


def initial(pop, dim):
    X = np.zeros((pop, dim))
    X[:,0] = np.random.random(pop) * (450- 250) + 250
    X[:,1] = np.random.random(size=(pop)) * (200 - 10) + 10
    X[:,2] = np.random.random(size=(pop)) * (5 - 0.5) + 0.5
    X[:,3] = np.random.random(size=(pop)) * (200 - 10) + 10
    X[:,4] = np.random.random(size=(pop)) * (2.1 - 0.3) + 0.3
    return X

'''边界检查函数'''


def BorderCheck(X, ub, lb):
    X[X[:,0]<250] = 250
    X[X[:,0]>450] = 450
    X[X[:,1]<10] = 10
    X[X[:,1]>200] = 200
    X[X[:,2]<0.5] = 0.5
    X[X[:,2]>5] = 5
    X[X[:,3]<10] = 10
    X[X[:,3]>200] = 200
    X[X[:,4]<0.3] = 0.3
    X[X[:,4]>2.1] = 2.1
    return X

def calc_e1(X):

    if X[0] < 250 or X[0]>450:
        return 5
    return 0


def calc_e2(X):

    if X[1] < 10 or X[1]>200:
        return 5
    return 0

def calc_e3(X):

    if X[2] < 0.5 or X[2]>5:
        return 5
    return 0


def calc_e4(X):
    if X[3] < 10 or X[3]>200:
        return 5
    return 0

def calc_e5(X):

    if X[4] < 0.3 or X[4]>2.1:
        return 5
    return 0


'''鲸鱼优化算法'''
def WOA(pop, dim, lb, ub):
    fitness = np.zeros((pop,1))
    fitness2 = np.zeros((pop,1))
    X= initial(pop, dim)  # 初始化种群
    for i in range(pop):  #混沌tent映射
        for j in range(dim):
            if X[i, j] >= 0 and X[i, j] < 0.7:
                X[i, j] = X[i, j] / 0.7
            elif X[i, j] >= 0.7 and X[i, j] <= 1:
                X[i, j] = (1 - X[i, j]) / 0.3


    for i in range(pop):
        e1 = calc_e1(X[i])
        e2 = calc_e2(X[i])
        e3 = calc_e3(X[i])
        e4 = calc_e4(X[i])
        e5 = calc_e5(X[i])
        while e1 > 0 or e2 > 0 or e3>0 or e4>0 or e5 >0:
            X[i, 0] = np.random.random() * (450 - 250) + 250
            X[i, 1] = np.random.random() * (200 - 10) + 10
            X[i, 2] = np.random.random() * (5 - 0.5) + 0.5
            X[i, 3] = np.random.random() * (200 - 10) + 10
            X[i, 4] = np.random.random() * (2.1 - 0.3) + 0.3
            e1 = calc_e1(X[i])
            e2 = calc_e2(X[i])
            e3 = calc_e3(X[i])
            e4 = calc_e4(X[i])
            e5 = calc_e5(X[i])

        fitness[i] = calc_f(X[i])



    GbestScore = fitness.max()
    GbestPositon = X[fitness.argmax()]
    best_fitness = []
    best_position = []
    best_position.append(GbestPositon)
    best_fitness.append(GbestScore)
    pbest = X.copy()
    for t in range(MaxIter):
        Leader = GbestPositon
        a = 2 - t * (2 / MaxIter)
        for i in range(pop):
            r1 = random.random()
            r2 = random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1
            l = 2 *random.random() -1

            for j in range(dim):
                p = random.random()
                if p < 0.5:
                    if np.abs(A) >= 1:
                        rand_leader_index = np.random.randint(0,pop)
                        X_rand = X[rand_leader_index, :]
                        D_X_rand = np.abs(C * X_rand[j] - X[i, j])
                        X[i, j] =X_rand[j] - A * D_X_rand
                    elif np.abs(A) < 1:
                        D_Leader = np.abs(C * Leader[j] - X[i, j])
                        X[i, j] = Leader[j] - A * D_Leader
                elif p >= 0.5:
                    distance2Leader = np.abs(Leader[j] - X[i, j])
                    X[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * math.pi) + Leader[j]



        X= BorderCheck(X, ub, lb)  # 边界检测

        for i in range(pop):
            e1 = calc_e1(X[i])
            e2 = calc_e2(X[i])
            e3 = calc_e3(X[i])
            e4 = calc_e4(X[i])
            e5 = calc_e5(X[i])
            while e1 > 0 or e2 > 0 or e3 > 0 or e4 > 0 or e5 > 0:
                X[i, 0] = np.random.random() * (450 - 250) + 250
                X[i, 1] = np.random.random() * (200 - 10) + 10
                X[i, 2] = np.random.random() * (5 - 0.5) + 0.5
                X[i, 3] = np.random.random() * (200 - 10) + 10
                X[i, 4] = np.random.random() * (2.1 - 0.3) + 0.3
                e1 = calc_e1(X[i])
                e2 = calc_e2(X[i])
                e3 = calc_e3(X[i])
                e4 = calc_e4(X[i])
                e5 = calc_e5(X[i])

            fitness2[i] = calc_f(X[i])


        for i in range(pop):
            if fitness2[i] > fitness[i]:
                fitness[i] = fitness2[i].copy()
                pbest[i] = X[i].copy()

        D = fitness.max()          #最优领域扰动
        Dp = pbest[fitness.argmax()].copy()
        r1 = np.random.random()
        r2 = np.random.random()
        if r2 < 0.5:
            Dp2 = Dp + 0.5 * r1 * Dp
            Dp2[0] = np.random.random() * (450 - 250) + 250
            Dp2[1] = np.random.random() * (200 - 10) + 10
            Dp2[2] = np.random.random() * (5 - 0.5) + 0.5
            Dp2[3] = np.random.random() * (200 - 10) + 10
            Dp2[4] = np.random.random() * (2.1 - 0.3) + 0.3
        else:
            Dp2 = Dp
        D2 = calc_f(Dp2)

        if D2 > D:
            pbest[fitness.argmax()] = Dp2
            fitness[fitness.argmax()] = D2



        GbestScore = fitness.max()
        GbestPositon = pbest[fitness.argmax()].copy()
        best_fitness.append(GbestScore)
        best_position.append(GbestPositon)
    return  GbestScore , GbestPositon,best_fitness,best_position


'''主函数 '''
# 设置参数
pop = 50  # 种群数量
MaxIter = 100  # 最大迭代次数
dim = 5 # 维度
lb = 0  # 下边界
ub = 450   # 上边界

best_score,best_position,  Curve ,position= WOA(pop, dim, lb, ub)
Curve = np.array(Curve)
best_position = np.array(best_position)
print('最优适应度值%.5f'% best_score)
#print('最优适应度值%.5f'% GbestScore.round(5))
print('迭代最优变量是' , best_position)
# 绘制适应度曲线
plt.figure(1)
plt.plot(Curve, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title('WOA', fontsize='large')

# 绘制搜索空间

plt.show()