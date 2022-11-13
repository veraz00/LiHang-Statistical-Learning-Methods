# https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/Logistic_and_maximum_entropy_models/logisticRegression.py
import numpy as np
import time
def load_data(filename):
    datalist, labellist = [], []
    fr = open(filename)
    for line in fr.readlines():
        list1 = line.strip().split(',')
        if int(list1[0]) == 0:
            labellist.append(0)
        else:
            labellist.append(1)
        datalist.append([int(i) for i in list1[1:]])
    return datalist, labellist

def train_model(traindatalist, trainlabellist, iter = 100):
    for i in range(len(trainlabellist)):
        # print('18', traindatalist[i])
        traindatalist[i].append(1)
    traindatalist = np.array(traindatalist)
    w = np.zeros(traindatalist.shape[1])
    h = 0.001
    for i in range(iter):
        for j in range(traindatalist.shape[0]):
        
            x = traindatalist[j]
            y = trainlabellist[j]
            wx = np.dot(w, x)
            w += x * y - x * np.exp(wx)/(1+np.exp(wx))
    return w

def model_test(testdatalist, testlabellist, w):
    errorCount = 0
    for i in range(len(testlabellist)):
        testdatalist[i].append(1)
        testdatalist[i] = np.array(testdatalist[i])
        wx = np.dot(w, testdatalist[i])
        predict = np.argmax([1, np.exp(wx)])
        if predict != testlabellist[i]:
            errorCount += 1
    return 1- errorCount/len(testlabellist)

if __name__ == '__main__':
    start = time.time()
    traindatalist, trainlabellist = load_data('D:\zenglinlin\data\mnist\mnist_train.csv')
    testdatalist, testlabellist = load_data('D:\zenglinlin\data\mnist\mnist_test.csv')
    
    w = train_model(traindatalist, trainlabellist, iter = 5)
    acc = model_test(testdatalist, testlabellist, w)
    print('acc', acc)
    print('time span: ', time.time() - start, 's')
    
# acc 0.902
# time span:  13.047762870788574 s