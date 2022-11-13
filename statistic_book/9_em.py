# EM MODEL explanation: https://www.youtube.com/watch?v=mYNmWKTsOxo&list=PLyAft-JyjIYpno8IfZZS0mnxD5TYZ6BIc&index=6
# em gaussian code: https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/EM/EM.py

import numpy as np
import random 
import math
import time 

def loadData(mu0, sigma0, mu1, sigma1, alpha0, alpha1):
    length = 100
    data0 = np.random.normal(mu0, sigma0, int(length * alpha0))
    data1 = np.random.normal(mu1, sigma1, int(length * alpha1))
    
    dataSet = list(data0) + list(data1)
    random.shuffle(dataSet)
    return dataSet

def EM_Train(dataSetList, iter = 200):
    dataSetArr = np.array(dataSetList)
    alpha0 = alpha1 = 0.5
    mu0, mu1 = 10, -50  # u average
    sigma0, sigma1 = 20, 10  # variance math.sqrt()
    step = 0
    while step < iter:
        print('step', step)
        gamma0, gamma1 = E_step(dataSetArr, alpha0, mu0, sigma0, alpha1, mu1, sigma1)  # gamm0: in model0, the weight of sample0, sample1, sample2
        mu0, mu1, sigma0, sigma1, alpha0, alpha1 = M_step(mu0, mu1, gamma0, gamma1, dataSetArr)
        print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f'%(
        alpha0, mu0, sigmod0, alpha1, mu1, sigmod1
    ))
        step += 1
    return alpha0, mu0, sigma0, alpha1, mu1, sigma1

def calcGauss(dataSetArr, mu, sigma):  # get N(mu, sigma) for x 
    result = 1/(sigma * np.sqrt(2*math.pi)) * np.exp(-1 * np.power(dataSetArr - mu, 2)/(2*sigma**2))
    return result    
    
def E_step(dataSetArr, alpha0, mu0, sigma0, alpha1, mu1, sigma1):
    gamma0 = alpha0 * calcGauss(dataSetArr, mu0, sigma0)
    gamma1 = alpha1 * calcGauss(dataSetArr, mu1, sigma1)
    
    sum = gamma0 + gamma1
    gamma0 = gamma0/sum
    gamma1 = gamma1/sum
    return gamma0, gamma1

def M_step(mu0, mu1, gamma0, gamma1, dataSetArr):
    mu0_new = np.dot(gamma0, dataSetArr)/sum(gamma0)
    mu1_new = np.dot(gamma1, dataSetArr)/sum(gamma1)
    
    sigma0_new =np.sqrt(np.dot(gamma0, (dataSetArr-mu0_new)**2) / sum(gamma0))
    sigma1_new =np.sqrt(np.dot(gamma1, (dataSetArr-mu1_new)**2)/sum(gamma1))
    alpha0_new = np.sum(gamma0) / len(gamma0)
    alpha1_new = np.sum(gamma1) / len(gamma1)

    #将更新的值返回
    return mu0_new, mu1_new, sigma0_new, sigma1_new, alpha0_new, alpha1_new

if __name__ == '__main__':
    start = time.time()

    #设置两个高斯模型进行混合，这里是初始化两个模型各自的参数
    #见“9.3 EM算法在高斯混合模型学习中的应用”
    # alpha是“9.3.1 高斯混合模型” 定义9.2中的系数α
    # mu0是均值μ
    # sigmod是方差σ
    #在设置上两个alpha的和必须为1，其他没有什么具体要求，符合高斯定义就可以
    alpha0 = 0.3; mu0 = -2; sigmod0 = 0.5
    alpha1 = 0.7; mu1 = 0.5; sigmod1 = 1

    #初始化数据集
    # dataSetList = loadData(mu0, sigmod0, mu1, sigmod1, alpha0, alpha1)
    dataSetList = [-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]

    #打印设置的参数
    print('---------------------------')
    print('the Parameters set is:')
    print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f'%(
        alpha0, mu0, sigmod0, alpha1, mu1, sigmod1
    ))

    #开始EM算法，进行参数估计
    alpha0, mu0, sigmod0, alpha1, mu1, sigmod1 = EM_Train(dataSetList)

    #打印参数预测结果
    print('----------------------------')
    print('the Parameters predict is:')
    print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f' % (
        alpha0, mu0, sigmod0, alpha1, mu1, sigmod1
    ))

    #打印时间
    print('----------------------------')
    print('time span:', time.time() - start)