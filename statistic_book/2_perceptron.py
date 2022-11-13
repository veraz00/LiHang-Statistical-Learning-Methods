# use mnist dataset; and use perceptron to achieve binary classification
import numpy as np
def loaddata(filename):
    print('start reading data')
    dataarr, labelarr = [], []
    fr = open(filename, 'r')
    for line in fr.readlines():
        curline = line.strip().split(',')
        if int(curline[0]) >= 5:
            labelarr.append(1)
        else:
            labelarr.append(-1)
        dataarr.append([int(num)//255 for num in curline[1::]])
    return np.array(dataarr), np.array(labelarr)

def perceptron(dataarr, labelarr, iter = 50):
    # import random
    # random.randrange(0, 10)
    # np.random.seed(10)
    # np.random.uniform(size = 3, low = 0, high = 10)   # generate float
    w, b = np.array([0.0] * len(dataarr[0])), 0
    for _ in range(iter):
        for num in range(dataarr.shape[0]):
            num = np.random.randint(0, len(labelarr)-1)
            if (np.dot(w.reshape(1, -1), dataarr[num].reshape(-1, 1)) + b ) * labelarr[num] <= 0:
                w += 0.001 * labelarr[num] * dataarr[num]
                b += 0.001 * labelarr[num]
    return w, b

def model_test(dataarr, labelarr, w, b):
    m, n = dataarr.shape
    err_count = 0
    for i in range(m):
        result = (np.dot(w.reshape(1, -1), dataarr[i, :].reshape(-1, 1)) + b) * labelarr[i] 
        if result <= 0:
            err_count += 1
    return 1-(err_count/m)

if __name__ == '__main__':
    dataarr, labelarr = loaddata('D:\zenglinlin\data\mnist\mnist_train.csv')
    print('dataarr.shape', dataarr.shape)
    test_data, test_label = loaddata('D:\zenglinlin\data\mnist\mnist_test.csv')
    w, b = perceptron(dataarr, labelarr)
    acc = model_test(test_data, test_label, w, b)
    print('acc', acc)
            
        
        
        
    