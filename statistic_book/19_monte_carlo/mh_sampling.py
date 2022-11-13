# https://blog.csdn.net/Changxing_J/article/details/118756757?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.pc_relevant_paycolumn_v3&spm=1001.2101.3001.4242.1&utm_relevant_index=3
# 常用概率分布（原生Python实现）

from abc import ABC 
from abc import abstractmethod 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BaseDistribution(ABC):
    def pdf(self, x):
        pass 
    def cdf(self, x):
        raise ValueError('Not define the distribution function')

class UniformDistribution(BaseDistribution):
    def __init__(self, a, b):
        self.a = a 
        self.b = b
    
    def pdf(self, x):
        if self.a < x and x< self.b:
            return 1/(self.b-self.a)
        else:
            return 0
    
    def cdf(self, x):
        if x < self.a:
            return 0
        elif self.a <= x and x < self.b:
            return (x-self.a)/(self.b-self.a)
        else:
            return 1

class GaussianDistribution(BaseDistribution):
    def __init__(self, u, s):
        self.u = u
        self.s = s 
    
    def pdf(self, x):
        return 1 /(np.sqrt(2 * np.pi)* self.s) * pow(np.e, -1/2 * (pow((x-self.u)/self.s, 2)))

# 直接抽样法（原生Python实现）
def direct_sampling_method(distribution, n_samples, a = -1e5, b = 1e5, tol = 1e-6, random_state = 0):
    np.random.seed(random_state)
    samples = []
    for _ in range(n_samples):
        y = np.random.rand()
        l, r= a, b
        while r -l > tol:
            m = (l+r)/2
            if distribution.cdf(m)>y:
                r = m
            else:
                l = m
        samples.append((l+r)/2)
    return samples


# 接受-拒绝抽样法（原生Python实现）
def accept_reject_sampling_method(d1, d2, c, n_samples, a = -1e5, b = 1e5, tol = 1e-6, random_state = 0):
    """接受-拒绝抽样法
    :param d1: 目标概率分布
    :param d2: 建议概率分布
    :param c: 参数c
    :param n_samples: 样本数
    :param a: 建议概率分布定义域左侧边界
    :param b: 建议概率分布定义域右侧边界
    :param tol: 容差
    :param random_state: 随机种子
    :return: 随机样本列表
    """
    np.random.seed(random_state)
    samples = []
    waiting = direct_sampling_method(d2, n_samples*2, a, b, tol, random_state)
    while len(samples) < n_samples:
        if not waiting:
            waiting = direct_sampling_method(d2, (n_samples-len(samples)))
        x = waiting.pop()
        u = np.random.rand()
        if u <= (d1.pdf(x)/c*d2.pdf(x)):
            samples.append(x)
        # else:
        #     print(u)
    return samples


def get_stationary_distribution(P, tol = 1e-8, max_iter = 1000):
    """迭代法求离散有限状态马尔可夫链的某个平稳分布
    
    根据平稳分布的定义求平稳分布。如果有无穷多个平稳分布，则返回其中任意一个。如果不存在平稳分布，则无法收敛。

    :param P: 转移概率矩阵
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :retun
    """
    n_components = len(P)
    pi0 = np.array([1/n_components] * n_components)
    print('pi0', pi0)
    for _ in range(max_iter):
        pi1 = np.dot(P, pi0)  # [0.5, 0. 0.5] * 2 = 2  # 2 is 
        if np.sum(np.abs(pi0-pi1)) < tol:
            break
        pi0 = pi1
    return pi0

# 判断马尔可夫链是否可约  
def is_reducible(P):
    n_components = len(P)
    for k in range(n_components):  # 检查从状态k出发能否到达任意状态
        visited = set()
        find = False
        stat0 = (False,) * k + (True,)+ (False,) * (n_components-k-1)
        while stat0 not in visited:
            visited.add(stat0)
            stat1 = [False] * n_components
            for j in range(n_components):
                if stat0[j] is True:
                    for i in range(n_components):
                        if P[i][j] > 0:
                            stat1[i] = True
            for i in range(k):
                if stat1[i] == True:  # 如果已经到达之前已检查可到达任意状态的状态，则不再继续寻找
                    find = True
                    break 
            if all(stat1) == True:
                find = True
                break
            stat0 = tuple(stat1)
        if not find:
            return True

    return False  # not reducible == can reach any state


# 计算马尔可夫链是否有周期性
from collections import Counter 
def is_periodic(P):
    n_components = len(P)
    P0 = P.copy()
    hash_P = tuple(P0.flat)  # hash_P (0.5, 0.5, 0.25, 0.25, 0.0, 0.25, 0.25, 0.5, 0.5)
    gcd = [0] * n_components
    visited = Counter()
    t = 1

    # 不断遍历时刻t，直至满足如下条件：当前t步转移矩阵之前已出现过2次（至少2次完整的循环）
    while visited[hash_P] < 2:
        visited[hash_P] += 1
        for i in range(n_components):
            if P0[i][i] > 0:
                if gcd[i] == 0:
                    gcd[i] = t
                else:
                    gcd[i] = np.gcd(gcd[i], t)  # 计算最大公约数
        for i in range(n_components):
            if gcd[i] == 0 or gcd[i] > 1:  
        # 检查当前时刻是否还有未返回(gcd[i]=0)/返回状态的所有时间长的最大公因数大于1(gcd[i]>1)的状态
                print(gcd, gcd[i])
                break  
        else:
            return False  # if all == 1
        P1 = np.dot(P0, P)
        P0 = P1
        hash_P = tuple(P0.flat)
        t += 1
    return True
  
# Metropolis-Hastrings算法
def metropolis_hasting_method(d1, func, n_features, m, n, x0, random_state = 0):
    """Metroplis-Hastings算法抽取样本

    :param d1: 目标概率分布的概率密度函数
    :param func: 目标求均值函数
    :param n_features: 随机变量维度
    :param x0: 初值（定义域中的任意一点即可）
    :param m: 收敛步数  #reach stable state
    :param n: 迭代步数
    :param random_state: 随机种子
    :return: 随机样本列表,随机样本的目标函数均值
    """
    np.random.seed(random_state)
    samples = []
    sum_ = 0
    for k in range(n):
        x1 = np.random.multivariate_normal(x0, np.diag([1] * n_features), 1)[0]  # [ 0.59 -0.18]
     
        # np.random.multivariate_normal(mean=mean, cov=conv, size=N)
        a = min(1, d1(x1)/d1(x0))
        u = np.random.rand()
        if u <= a:      # 若u<=a，则转移状态；否则不转移
            x0 = x1

        if k >= m: 
            samples.append(x0)
            sum_ += func(x0)
    return samples, sum_/(n-m)


if __name__== '__main__':
    # distribution = UniformDistribution(-3, 3)
    # samples = direct_sampling_method(distribution, 10, -3, 3)
    # print('samples from direct sampling: ', samples)

    # d1 = GaussianDistribution(0, 1)
    # d2 = UniformDistribution(-3, 3)
    # c = (1/np.sqrt(2*np.pi)) / (1/6)
    # samples = accept_reject_sampling_method(d1, d2, c, 2, -3, 3)
    # print('samples from accept_rejecting: ', samples)

    np.set_printoptions(precision=2, suppress=True)
    P = np.array([[0.5, 0.5, 0.25],
                  [0.25, 0, 0.25],
                  [0.25, 0.5, 0.5]])
    print(get_stationary_distribution(P))  # [0.4 0.2 0.4]
    # print(is_reducible(P))  # False
    # print(is_periodic(P))

    def d1_pdf(x):
        return x[0] * pow(np.e, -x[1]) if 0 < x[0] and x[0] < x[1] else 0
    
    def f(x):
        return x[0] + x[1]
    samples, avg = metropolis_hasting_method(d1_pdf, f, 2, m=1000, n=11000, x0=[5, 8])

    def draw_distribution():
        X, Y = np.meshgrid(np.arange(0, 10, 0.1), np.arange(0, 10, 0.1))
        Z = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                Z[i][j] = d1_pdf([i/10, j/10])
        fig = plt.figure()
        # plt.imshow(Z, cmap = 'rainbow')
        # plt.colorbar()
        # plt.show()
        ax = Axes3D(fig)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
        plt.show()


    def draw_sample():
        """绘制样本概率密度函数的图"""
        X, Y = np.meshgrid(np.arange(0, 10, 0.1), np.arange(0, 10, 0.1))
        Z = np.zeros((100, 100))
        for i, j in samples:
            if i < 10 and j < 10:
                Z[int(i // 0.1)][int(j // 0.1)] += 1

        fig = plt.figure()
        # plt.imshow(Z, cmap="rainbow")
        # plt.colorbar()
        # plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
        plt.show()

    draw_distribution()
    draw_sample()