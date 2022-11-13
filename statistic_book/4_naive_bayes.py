# https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/NaiveBayes/NaiveBayes.py
import numpy as np
import time 
def loadData(filename):
    dataarr, labelarr = [], []
    fr = open(filename)  # read csv as a file 
    for line in fr.readlines():
        curline =line.strip().split(',')
        dataarr.append([1*(int(i)>128) for i in curline[1:]])
        labelarr.append(int(curline[0]))  
    return np.array(dataarr), np.array(labelarr)  # 1024 * 784, 1024

def get_all_probability(traindataarr, trainlabelarr):
    classnum = max(trainlabelarr) + 1
    py = np.zeros(classnum)
    for i in range(classnum):
        py[i] =(1+ np.sum((trainlabelarr == i)* 1))/(classnum + len(trainlabelarr))
        py[i] = np.log(py[i])
    featurenum = len(traindataarr[0])
    px_y = np.zeros((classnum, featurenum, 2))
    for i in range(len(trainlabelarr)):
        label = trainlabelarr[i]
        x = traindataarr[i]
        for f in range(featurenum):
            px_y[label][f][x[f]] += 1 
    
    for label in range(classnum):
        for j in range(featurenum):
            px_y0 = px_y[label][j][0]
            px_y1= px_y[label][j][1]
            px_y[label][j][0] = np.log((px_y0+1)/(px_y0+px_y1+2))
            px_y[label][j][1] = np.log((px_y1+1)/(px_y0+px_y1+2))
    return py, px_y

def NaiveBayes(py, px_y, x):
    featurenum = len(x)
    classnum = 10
    result_list = []

    for c in range(classnum):
        result = 1
        for f in range(featurenum):
            result += px_y[c][f][x[f]]
        result += py[c]
        result_list.append(result)
    return np.argmax(result_list)


def model_test(py, px_y, testdataarr, testlabelarr):
    error = 0
    for i in range(len(testdataarr)):
        predict = NaiveBayes(py, px_y, testdataarr[i])
        if predict != testlabelarr[i]:
            error += 1
    return 1-(error / len(testdataarr))

if __name__ == '__main__':
    start = time.time()
    
    print('start read trainsSet')
    trainDataArr, trainLabelArr = loadData('D:\zenglinlin\data\mnist\mnist_train.csv')

    # 获取测试集
    print('start read testSet')
    testDataArr, testLabelArr = loadData('D:\zenglinlin\data\mnist\mnist_test.csv')

    #开始训练，学习先验概率分布和条件概率分布
    print('start to train')
    Py, Px_y = get_all_probability(trainDataArr, trainLabelArr)
    print('py', Py)
    print('px_y', Px_y)
    
    #使用习得的先验概率分布和条件概率分布对测试集进行测试
    print('start to test')
    accuracy = model_test(Py, Px_y, testDataArr, testLabelArr)

    #打印准确率
    print('the accuracy is:', accuracy)
    #打印时间
    print('time span:', time.time() -start)