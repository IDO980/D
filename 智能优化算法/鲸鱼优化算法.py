import numpy as np
import random
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''优化函数'''



def calc_f(X):
    x = X[0]
    y = X[1]
    return x **2 + y**2 -4 * x + 4


def initial(pop, dim):

    X = np.random.random(size = (pop,dim)) * (ub - lb) + lb

    return X


'''边界检查函数'''


def BorderCheck(X, ub, lb):
    X[X<lb] = lb
    X[X>ub] = ub
    return X

def calc_e1(X):
    """计算第一个约束的惩罚项"""
    e = -X[0] + X[1]  -2
    return e

def calc_e2(X):
    """计算第二个约束的惩罚项"""
    return X[0]**2 - X[1] + 1

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
        while e1 > 0 or e2 > 0:
            X[i] = np.random.random((1,dim)) * (ub - lb) + lb
            e1 = calc_e1(X[i])
            e2 = calc_e2(X[i])
        fitness[i] = calc_f(X[i])



    GbestScore = fitness.min()
    GbestPositon = X[fitness.argmin()]
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
            while e1 > 0 or e2 > 0:
                X[i] = np.random.random((1, dim)) * (ub - lb) + lb
                e2 = calc_e2(X[i])
                e1 = calc_e1(X[i])
            fitness2[i] = calc_f(X[i])


        for i in range(pop):
            if fitness2[i] < fitness[i]:
                fitness[i] = fitness2[i].copy()
                pbest[i] = X[i].copy()

        D = fitness.min()          #最优领域扰动
        Dp = pbest[fitness.argmin()].copy()
        r1 = np.random.random()
        r2 = np.random.random()
        if r2 < 0.5:
            Dp2 = Dp + 0.5 * r1 * Dp
        else:
            Dp2 = Dp
        D2 = calc_f(Dp2)

        if D2 < D:
            pbest[fitness.argmin()] = Dp2
            fitness[fitness.argmin()] = D2



        GbestScore = fitness.min()
        GbestPositon = pbest[fitness.argmin()].copy()
        best_fitness.append(GbestScore)
    #return GbestScore, GbestPositon, best_fitness,g_e
    return  GbestScore , GbestPositon,best_fitness


'''主函数 '''
# 设置参数
pop = 100  # 种群数量
MaxIter = 50  # 最大迭代次数
dim = 2 # 维度
lb = 0  # 下边界
ub = 5   # 上边界

best_score,best_position,  Curve = WOA(pop, dim, lb, ub)
Curve = np.array(Curve)
best_position = np.array(best_position)
print('最优适应度值%.5f'% best_score)
#print('最优适应度值%.5f'% GbestScore.round(5))
print('迭代最优变量是：x=%.5f, y=%.5f' % (best_position[0],(best_position[1])))
# print('迭代约束惩罚项是：', best_e[Curve.argmin()])
# 绘制适应度曲线
plt.figure(1)
plt.plot(Curve, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title('WOA', fontsize='large')

# 绘制搜索空间

plt.show()