import numpy as np
import pylab as plt

gy_size = 100    #雇佣蜂
gc_size = 100   #观察蜂
dim = 2
limit = np.round(0.6 * dim * gy_size)
max_iter = 200
ub = 10
lb = -10


best_fitness = []
best_best = []

def calc_f(X):
    """计算粒子的的适应度值，也就是目标函数值，X 的维度是 size * 2 """
    A = 10
    pi = np.pi
    x = X[0]
    y = X[1]
    return 2 * A + x ** 2 - A * np.cos(2 * pi * x) + y ** 2 - A * np.cos(2 * pi * y)



def find(r,p):
    for m in range(len(p)):
        if r <= p[m]:
            return m



g_fitness = np.inf
gbest = np.zeros(2)
pop = (ub - lb) * np.random.random(size = (gy_size,dim)) + lb  #种群初始化

p_fitness = np.zeros((gy_size,1))

for i in range(gy_size):
    p_fitness[i,:] = calc_f(pop[i,:])


g_fitness = p_fitness.min()
gbest = pop[p_fitness.argmin()]
L = [0 for i in range(gy_size)]
best_fitness.append(g_fitness)

for t in range(max_iter):
    # 雇佣蜂阶段
    for i in range(gy_size):
        k = np.random.randint(0,gy_size)
        while k == i :
            k = np.random.randint(0, gy_size)
        # fai = np.random.random() *2 -1
        fai = np.random.random()  - 0.5
        new_X = pop[i,:] + fai * (pop[i,:] - pop[k,:])
        new_X[new_X > ub] = ub
        new_X[new_X < lb] = lb

        new_X = np.array(new_X)
        new_cost = calc_f(new_X)

        if new_cost <= p_fitness[i]:
            p_fitness[i] = new_cost
            pop[i] = new_X
        elif new_cost > p_fitness[i]:
            L[i] = L[i] + 1

    m_fitness = p_fitness.mean()
    F = np.zeros(gy_size)
    for i in range(gy_size):
        F[i] = np.exp(-1 * p_fitness[i]/m_fitness)
    P = np.cumsum(F/np.sum(F))


    #下一阶段
    for i in range(gc_size):
        r = np.random.random()
        j = find(r,P)
        k = np.random.randint(0, gy_size)
        while k == j:
            k = np.random.randint(0, gy_size)
        fai = np.random.random() *2 -1
        new_X = pop[j] + fai * (pop[j] - pop[k])
        new_X[new_X > ub] = ub
        new_X[new_X < lb] = lb

        new_X = np.array(new_X)
        new_cost = calc_f(new_X)

        if new_cost <= p_fitness[j]:
            p_fitness[j] = new_cost
            pop[j] = new_X
        elif new_cost > p_fitness[j]:
            L[j] = L[j] + 1



    for i in range(gy_size):
        if L[i] >= limit:
            pop[i] = (ub - lb) * np.random.random(size = (1,dim)) + lb
            p_fitness[i] = calc_f(pop[i])
            L[i] = 0



    for i in range(gy_size):
        if p_fitness[i,:] < g_fitness:
            g_fitness = p_fitness[i,:].copy()
            gbest = pop[i].copy()

    best_fitness.append(g_fitness)

print('------------------------')
print(g_fitness.round(5))
print(gbest.round(4))
plt.plot(range(max_iter+1),best_fitness)
plt.show()