import numpy as np
import random
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''优化函数'''



def calc_f(X):
    A = 10
    pi = np.pi
    x = X[0]
    y = X[1]
    return 2 * A + x ** 2 - A * np.cos(2 * pi * x) + y ** 2 - A * np.cos(2 * pi * y)

''' 种群初始化函数 '''


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
    e = X[0] + X[1] + 2
    return max(0, e)

def calc_e2(X):
    """计算第二个约束的惩罚项"""
    e = 3 * X[0] - 2 * X[1] +4
    return max(0, e)

def calc_Lj(e1, e2):
    """根据每个粒子的约束惩罚项计算Lj权重值，e1, e2列向量，表示每个粒子的第1个第2个约束的惩罚项值"""
    # 注意防止分母为零的情况
    if (e1.sum() + e2.sum()) <= 0:
        return 0, 0
    else:
        L1 = e1.sum() / (e1.sum() + e2.sum())
        L2 = e2.sum() / (e1.sum() + e2.sum())
    return L1, L2

def update_pbest(pbest, pbest_fitness, pbest_e, xi, xi_fitness, xi_e):

    if pbest_e <= 0.0000001 and xi_e <= 0.0000001:
        if pbest_fitness <= xi_fitness:
            return pbest, pbest_fitness, pbest_e,False
        else:
            return xi, xi_fitness, xi_e,True
    # 规则2，如果当前位置违反约束而历史最优没有违反约束，则取历史最优
    if pbest_e < 0.0000001 and xi_e >= 0.0000001:
        return pbest, pbest_fitness, pbest_e,False
    # 规则3，如果历史位置违反约束而当前位置没有违反约束，则取当前位置
    if pbest_e >= 0.0000001 and xi_e < 0.0000001:
        return xi, xi_fitness, xi_e,True
    # 规则4，如果两个都违反约束，则取适应度值小的
    if pbest_fitness <= xi_fitness:
        return pbest, pbest_fitness, pbest_e,False
    else:
        return xi, xi_fitness, xi_e,True



'''计算适应度函数'''




'''适应度排序'''



'''鲸鱼优化算法'''
def WOA(pop, dim, lb, ub):
    limit = MaxIter/5
    L = np.zeros((pop,1))
    value = np.zeros((pop,1))
    e1 = np.zeros((pop,1))
    e2 = np.zeros((pop,1))
    X= initial(pop, dim)  # 初始化种群
    for i in range(pop):
        for j in range(dim):
            if X[i,j] >=0 and X[i,j] <0.7:
                X[i,j] = X[i,j]/0.7
            elif X[i,j]>=0.7 and X[i,j]<=1:
                X[i,j] = (1-X[i,j])/0.3
    for i in range(pop):
        value[i] = calc_f(X[i])
        e1[i] = calc_e1(X[i])
        e2[i] = calc_e2(X[i])
    L1,L2 = calc_Lj(e1,e2)
    p_e =L1 * e1 +L2 * e2
    fitness =value + p_e

    GbestScore = fitness.min()
    GbestPositon = X[fitness.argmin()]
    g_e = p_e[fitness.argmin()]

    best_fitness = []
    best_position = []
    best_e = []
    best_position.append(GbestPositon)
    best_e.append(g_e)
    best_fitness.append(GbestScore)
    pbest = X
    for t in range(MaxIter):
        k = 0.6
        d = 0.6
        Leader = GbestPositon

        w = (0.9 - 0.4) * (0.8 *(1-(t/MaxIter)**k)) + 0.4
        a = -np.cos(np.pi * t/MaxIter + np.pi) + 1
        for i in range(pop):
            r1 = random.random()
            r2 = random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1
            l = 2 *random.random() -1

            for j in range(dim):
                p = random.random()
                p1 = 1 - np.log10(1 + (9*t)/MaxIter)
                # if p < 0.5:
                if p < p1:
                    if np.abs(A) >= 1:
                        rand_leader_index = np.random.randint(0,pop)
                        X_rand = X[rand_leader_index, :]
                        D_X_rand = np.abs(C * X_rand[j] - X[i, j])
                        X[i, j] =w*X_rand[j] - A * D_X_rand
                    elif np.abs(A) < 1:
                        D_Leader = np.abs(C * Leader[j] - X[i, j])
                        X[i, j] = w*Leader[j] - A * D_Leader
                elif p >= p1:
                    distance2Leader = np.abs(Leader[j] - X[i, j])
                    X[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * math.pi) + w* Leader[j]



        X= BorderCheck(X, ub, lb)  # 边界检测
        for i in range(pop):
            value[i] = calc_f(X[i])
            e1[i] = calc_e1(X[i])
            e2[i] = calc_e2(X[i])
        L1, L2 = calc_Lj(e1, e2)
        p_e2 = t * np.sqrt(t)*(L1 * e1 +L2 * e2)
        fitness2 = value + p_e2
        for i in range(pop):
            pbesti_ = pbest[i]
            p_fitness_ = fitness[i]
            p_e_ = p_e[i]
            xi = X[i]
            xi_fitness = fitness2[i]
            xi_e = p_e2[i]
            # 计算更新个体历史最优
            pbest__, p_fitness__, p_e__ ,flag= update_pbest(pbesti_, p_fitness_, p_e_, xi, xi_fitness, xi_e)
            pbest[i] = pbest__
            fitness[i] = p_fitness__
            p_e[i] = p_e__
            if flag == False:
                L[i] += 1

        for i in range(pop):
            if L[i] >= limit:
                X[i] = (ub - lb) * np.random.random(size=(1, dim)) + lb
                fitness[i] = calc_f(X[i])
                L[i] = 0

        GbestScore = fitness.min()
        GbestPositon = pbest[fitness.argmin()]
        g_e = p_e[fitness.argmin()]
        best_fitness.append(GbestScore)
        best_position.append(GbestPositon)
        best_e.append(g_e)
    #return GbestScore, GbestPositon, best_fitness,g_e
    return best_position,best_e,best_fitness


'''主函数 '''
# 设置参数
pop = 100  # 种群数量
MaxIter = 1000  # 最大迭代次数
dim = 2 # 维度
lb = -5  # 下边界
ub = 5   # 上边界

best_position, best_e, Curve = WOA(pop, dim, lb, ub)
Curve = np.array(Curve)
best_position = np.array(best_position)
best_e = np.array(best_e)
print('最优适应度值%.5f'% Curve.min())

#print('最优适应度值%.5f'% GbestScore.round(5))
print('迭代最优变量是：x=%.5f, y=%.5f' % (best_position[Curve.argmin()][0],(best_position[Curve.argmin()][1])))
print('迭代约束惩罚项是：', best_e[Curve.argmin()])
# 绘制适应度曲线
plt.figure(1)
plt.plot(Curve, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title('WOA', fontsize='large')

# 绘制搜索空间

plt.show()