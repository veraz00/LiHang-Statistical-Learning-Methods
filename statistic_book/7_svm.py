def loadData(filename):
    data, label = [], []
    f = open(filename)
    for line in f.readlines():
        list1 = line.strip().split(',')
        if list1[0] == '0':
            label.append(1)
        else:
            label.append(-1)
        data1 = [int(i)/225 for i in list1[1:]]
        data.append(data1)
    return data, label
import numpy as np
import cvxopt

class SVM:
    def __init__(self, traindatalist, trainlabellist, sigma = 10, C=1, power=4, gamma=None, coef=4):
        '''
        SVM相关参数初始化
        :param trainDataList:训练数据集
        :param trainLabelList: 训练测试集
        :param sigma: 高斯核中分母的σ
        :param C:软间隔中的惩罚参数
        :param toler:松弛变量
        '''
        self.train
        self.traindatamat = np.mat(traindatalist)
        self.trainlabelmat = np.mat(trainlabellist)
        self.m, self.n = np.shape(self.traindatamat)  # #m：训练集数量    n：样本特征数目
        self.C = C
        self.sigma = sigma
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None
        self.k = self.calckernel()  # #核函数（初始化时提前计算）
  
    
    def calckernel(self): # gaussian kernel function
        k = [[0 for i in range(self.m)]for j in range(self.m)] # m：训练集数量
        for i in range(self.m): 
            x = self.traindatamat[i, :]
            for j in range(i, self.m):
                z = self.traindatamat[j, :]
                result = np.dot(x-z, (x-z).T) 
                result = np.exp(-1 * result/(2*self.sigma**2))
                k[i][j] = result
                k[j][j] = result 
        return np.mat(k)

    def calcSinglKernel(self, x1, x2):
        '''
        单独计算核函数
        :param x1:向量1
        :param x2: 向量2
        :return: 核函数结果
        '''
        #按照“7.3.3 常用核函数”式7.90计算高斯核
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.sigma ** 2))
        #返回结果
        return np.exp(result)


    def train(self):
        
        if not self.gamma:
            self.gamma = 1/self.n
        P = cvxopt.matrix(np.dot(self.trainlabelmat.T, self.trainlabelmat) * self.k, tc = 'd')
        q = cvxopt.matrix(np.ones(self.m) * -1)
        A = cvxopt.matrix(y, (1, self.m), tc='d')
        b = cvxopt.matrix(0, tc='d')
        # cvxopt.matrix(value, size, tc)
            # tc stands for type code. The possible values are 'i', 'd', and 'z', 
            # for integer, real (double), and complex matrices, respectively.
        if not self.C:
            G = cvxopt.matrix(np.identity(self.n) * -1)     
# np.identity(3)
# array([[1.,  0.,  0.],
#        [0.,  1.,  0.],
#        [0.,  0.,  1.]])
            h = cvxopt.matrix(np.zeros(self.n))
        else:
            G_max = np.identity(self.n) * -1
            G_min = np.identity(self.n)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(self.n))
            h_min = cvxopt.matrix(np.ones(self.n) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))
        # Solve the quadratic optimization problem using cvxopt  # why is quadratic??
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])

            # https://stackoverflow.com/questions/32543475/how-python-cvxopt-solvers-qp-basically-works
        
        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = lagr_mult > 1e-7
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        # Get the samples that will act as support vectors
        self.support_vectors = self.traindatamat[idx, :]
        print('94', len(self.support_vectors))
        # Get the corresponding labels
        self.support_vector_labels = self.trainlabelmat[idx, 0]
        
            # Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.calcSinglKernel(self.support_vectors[i], self.support_vectors[0])
# calcSinglKernel(self, x1, x2):
    def predict(self, X, Y):

        # Iterate through list of samples and make predictions
        errorCnt = 0
        for i, sample in enumerate(X):
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.calcSinglKernel(self.support_vectors[i], sample)
            prediction += self.intercept
            if prediction != Y[i]:
                errorCnt += 1
            
            
        return 1-errorCnt/len(Y)

import time 
if __name__ == '__main__':
    start = time.time()

    print('start read trainsSet')
    trainDataList, trainLabelList = loadData('D:\zenglinlin\data\mnist\mnist_train.csv')

    # 获取测试集
    print('start read testSet')
    testDataList, testLabelList = loadData('D:\zenglinlin\data\mnist\mnist_test.csv')

       #初始化SVM类
    print('start init SVM')
        #初始化SVM类
    print('start init SVM')
    svm = SVM(trainDataList[:1000], trainLabelList[:1000])
    print('start to train')
    svm.train()

    # 开始测试
    print('start to test')
    accuracy = svm.test(testDataList[:100], testLabelList[:100])
    print('the accuracy is:%d'%(accuracy * 100), '%')
