import time 
import numpy as np
def loadData(fileName):
    '''
    加载文件
    :param fileName:要加载的文件路径
    :return: 数据集和标签集
    '''
    #存放数据及标记
    dataArr = []; labelArr = []
    #读取文件
    fr = open(fileName)
    #遍历文件中的每一行
    for line in fr.readlines():
        #获取当前行，并按“，”切割成字段放入列表中
        #strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
        #split：按照指定的字符将字符串切割成每个字段，返回列表形式
        curLine = line.strip().split(',')
        #将每行中除标记外的数据放入数据集中（curLine[0]为标记信息）
        #在放入的同时将原先字符串形式的数据转换为整型
        #此外将数据进行了二值化处理，大于128的转换成1，小于的转换成0，方便后续计算
        dataArr.append([int(int(num) > 128) for num in curLine[1:]])
        #将标记信息放入标记集中
        #放入的同时将标记转换为整型
        labelArr.append(int(curLine[0]))
    #返回数据集和标记
    return dataArr, labelArr

def majorClass(labelArr):
    classDict = {}
    for i in range(len(labelArr)):
        if labelArr[i] in classDict.keys():
            classDict[labelArr[i]] += 1
        else:
            classDict[labelArr[i]] = 1
    classsort = sorted(classDict.items(), key = lambda x: x[1], reverse = True) # key: value
    return classsort[0][0]  # classort[0] = {class1: 3}


def calc_H_D(trainLabelArr):
    H_D = 0
    trainLabelSet = set([label for label in trainLabelArr])
    for i in trainLabelSet:
        p = trainLabelArr[trainLabelArr==i].size/trainLabelArr.size
        H_D += -1* p* np.log2(p)
    return H_D


def calcH_D_A(trainDataArr_DevFeature, trainLabelArr):
    '''
    计算经验条件熵
    :param trainDataArr_DevFeature:切割后只有feature那列数据的数组
    :param trainLabelArr: 标签集数组
    :return: 经验条件熵
    '''
    #初始为0
    H_D_A = 0
    #在featue那列放入集合中，是为了根据集合中的数目知道该feature目前可取值数目是多少
    trainDataSet = set([label for label in trainDataArr_DevFeature])

    #对于每一个特征取值遍历计算条件经验熵的每一项
    for i in trainDataSet:
        #计算H(D|A)
        #trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size:|Di| / |D|
        #calc_H_D(trainLabelArr[trainDataArr_DevFeature == i]):H(Di)
        H_D_A += trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size \
                * calc_H_D(trainLabelArr[trainDataArr_DevFeature == i])  # trainDataArr_DevFeature == i]!!
    #返回得出的条件经验熵
    return H_D_A


def calcBestFeature(trainDataList, trainLabelList):
    trainDataArr = np.array(trainDataList)
    trainLabelArr = np.array(trainLabelList)
    featureNum = trainDataArr.shape[1]
    
    maxG_D_A = -1 # 最大信息增益
    maxFeature = -1   # 最大信息增益的特征
    H_D = calc_H_D(trainLabelArr)  # parents H_D
    for feature in range(featureNum):
        trainDataArr_DevideByFeature = np.array(trainDataArr[:, feature].flat)
        G_H_A = H_D - calcH_D_A(trainDataArr_DevideByFeature, trainLabelArr)
        if G_H_A > maxG_D_A:
            maxG_D_A = G_H_A
            maxFeature = feature
    return maxFeature, maxG_D_A


def getSubDataArr(trainDataArr, trainLabelArr, A, a):
    retDataArr, retLabelArr = [], []
    for i in range(len(trainDataArr)):
        if trainDataArr[i][A] == a:
            retDataArr.append(trainDataArr[i][0:A] + trainDataArr[i][A+1:])
            retLabelArr.append(trainLabelArr[i])
    print('81', len(retLabelArr))
    return retDataArr, retLabelArr
            
    

def createTree(*dataSet):  # retDataArr, retLabelArr
    Epsilon = 1e-2
    trainDataList = dataSet[0][0]
    trainLabelList = dataSet[0][1]
    print('start a node', len(trainDataList[0]), len(trainLabelList))
    # 始一个子节点创建，打印当前特征向量数目及当前剩余样本数目
    classDict = {i for i in trainLabelList}
    if len(classDict) == 1:
        return trainLabelList[0]
    if len(trainDataList) == 0:
        return majorClass(trainLabelList)
    
    Ag, EpsilonGet = calcBestFeature(trainDataList, trainLabelList)  # maxFeature, maxG_D_A
    print('81', Ag, EpsilonGet)
    if EpsilonGet < Epsilon:
        return majorClass(trainLabelList)
    treeDict = {Ag: {}}
    treeDict[Ag][0] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 0))  # Ag: feature, 0 is value
    treeDict[Ag][1]= createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 1))
    return treeDict

def predict(testDataList, tree):
    while True:
        (key, value) = tree.items()
        if type(tree[key]).__name__ == 'dict':
            dataVal = testDataList[key]
            del testDataList[key]
            tree = value[dataVal]
            if type(tree).__name__ == 'int':
                return tree
        else:
            return value
            
def model_test(testDataList, testLabelList, tree):
    errorCnt = 0
    for i in range(len(testDataList)):
        if testLabelList[i] != predict(testDataList[i], tree):
            errorCnt += 1
    return 1- errorCnt/len(testDataList)

if __name__ == '__main__':
    start = time.time()
    print('start read trainsSet')
    trainDataList, trainLabelList = loadData('D:\zenglinlin\data\mnist\mnist_train.csv')

    # 获取测试集
    print('start read testSet')
    testDataList, testLabelList = loadData('D:\zenglinlin\data\mnist\mnist_test.csv')
    
    print('start create tree')
    tree = createTree((trainDataList, trainLabelList))
    print('tree is:', tree)

    #测试准确率
    print('start test')
    accur = model_test(testDataList, testLabelList, tree)
    print('the accur is:', accur)

    #结束时间
    end = time.time()
    print('time span:', end - start)