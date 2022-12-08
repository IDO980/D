import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# PSO的参数
 # 惯性因子，一般取1
c1 = 2  # 学习因子，一般取2
c2 = 2  #
r1 = None  # 为两个（0,1）之间的随机数
r2 = None
dim = 2  # 维度的维度
size = 100  # 种群大小，即种群中小鸟的个数
iter_num = 1000  # 算法最大迭代次数
max_vel = 0.5  # 限制粒子的最大速度为0.5
fitneess_value_list = []  # 记录每次迭代过程中的种群适应度值变化


def calc_f(X):
    """计算粒子的的适应度值，也就是目标函数值，X 的维度是 size * 2 """
    A = 10
    pi = np.pi
    x = X[0]
    y = X[1]
    return 2 * A + x ** 2 - A * np.cos(2 * pi * x) + y ** 2 - A * np.cos(2 * pi * y)

def calc_e1(X):
    """计算第一个约束的惩罚项"""
    e = X[0] + X[1] +2
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



def velocity_update(V, X, pbest, gbest,t):
    """
    根据速度更新公式更新每个粒子的速度
    :param V: 粒子当前的速度矩阵，20*2 的矩阵
    :param X: 粒子当前的位置矩阵，20*2 的矩阵
    :param pbest: 每个粒子历史最优位置，20*2 的矩阵
    :param gbest: 种群历史最优位置，1*2 的矩阵
    """
    r1 = np.random.random((size, 1))
    r2 = np.random.random((size, 1))
    V = (0.9-0.5*t/iter_num) * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)  # 直接对照公式写就好了
    # 防止越界处理
    V[V < -max_vel] = -max_vel
    V[V > max_vel] = max_vel
    return V

def position_update(X, V):
    """
    根据公式更新粒子的位置
    :param X: 粒子当前的位置矩阵，维度是 20*2
    :param V: 粒子当前的速度举着，维度是 20*2
    """
    return X + V

def update_pbest(pbest, pbest_fitness, pbest_e, xi, xi_fitness, xi_e):
    """
    判断是否需要更新粒子的历史最优位置
    :param pbest: 历史最优位置
    :param pbest_fitness: 历史最优位置对应的适应度值
    :param pbest_e: 历史最优位置对应的约束惩罚项
    :param xi: 当前位置
    :param xi_fitness: 当前位置的适应度函数值
    :param xi_e: 当前位置的约束惩罚项
    :return:
    """
    # 下面的 0.0000001 是考虑到计算机的数值精度位置，值等同于0
    # 规则1，如果 pbest 和 xi 都没有违反约束，则取适应度小的
    if pbest_e <= 0.0000001 and xi_e <= 0.0000001:
        if pbest_fitness <= xi_fitness:
            return pbest, pbest_fitness, pbest_e
        else:
            return xi, xi_fitness, xi_e
    # 规则2，如果当前位置违反约束而历史最优没有违反约束，则取历史最优
    if pbest_e < 0.0000001 and xi_e >= 0.0000001:
        return pbest, pbest_fitness, pbest_e
    # 规则3，如果历史位置违反约束而当前位置没有违反约束，则取当前位置
    if pbest_e >= 0.0000001 and xi_e < 0.0000001:
        return xi, xi_fitness, xi_e
    # 规则4，如果两个都违反约束，则取适应度值小的
    if pbest_fitness <= xi_fitness:
        return pbest, pbest_fitness, pbest_e
    else:
        return xi, xi_fitness, xi_e




X = np.random.uniform(-5, 5, size=(size, dim))
# 初始化种群的各个粒子的速度
V = np.random.uniform(-0.5, 0.5, size=(size, dim))
p_value = np.zeros((size,1))
e1 = np.zeros((size,1))
e2 = np.zeros((size,1))


# 初始化粒子历史最优位置为当当前位置
pbest = X
# 计算每个粒子的适应度
for i in range(size):
    p_value[i] = calc_f(X[i])  # 目标函数值
    e1[i] = calc_e1(X[i])  # 第一个约束的惩罚项
    e2[i] = calc_e2(X[i])  # 第二个约束的惩罚项

L1,L2 = calc_Lj(e1,e2)
p_e = L1 * e1 + L2 * e2
p_fitness = p_value + p_e
g_fitness = p_fitness.min()
gbest = X[p_fitness.argmin()]
g_e = p_e[p_fitness.argmin()]
# 记录迭代过程的最优适应度值
fitneess_value_list.append(g_fitness)
# 接下来开始迭代
for j in range(iter_num):
    # 更新速度
    V = velocity_update(V, X, pbest=pbest, gbest=gbest,t = j)
    # 更新位置
    X = position_update(X, V)
    # 计算每个粒子的目标函数和约束惩罚项
    for i in range(size):
         p_value[i] = calc_f(X[i])  # 目标函数值
         e1[i] = calc_e1(X[i])  # 第一个约束的惩罚项
         e2[i] = calc_e2(X[i])  # 第二个约束的惩罚项
    L1, L2 = calc_Lj(e1, e2)
    p_e2 =j * np.sqrt(j)*( L1 * e1 + L2 * e2)
    p_fitness2 = p_value + p_e2

    for i in range(size):
        pbesti_ = pbest[i]
        p_fitness_ = p_fitness[i]
        p_e_ = p_e[i]
        xi = X[i]
        xi_fitness = p_fitness2[i]
        xi_e = p_e2[i]
        # 计算更新个体历史最优
        pbest__, p_fitness__, p_e__ = update_pbest(pbesti_, p_fitness_, p_e_, xi, xi_fitness, xi_e)
        pbest[i] = pbest__
        p_fitness[i] = p_fitness__
        p_e[i] = p_e__

    gbest = pbest[p_fitness.argmin()]
    g_fitness =p_fitness.min()
    g_e = p_e[p_fitness.argmin()]
    fitneess_value_list.append(g_fitness)

# 最后绘制适应度值曲线
print('迭代最优结果是：%.5f' % g_fitness)
print('迭代最优变量是：x=%.5f, y=%.5f' % (gbest[0], gbest[1]))
print('迭代约束惩罚项是：', g_e)


# 绘图
plt.plot(fitneess_value_list[: 30], color='r')
plt.title('迭代过程')
plt.show()
