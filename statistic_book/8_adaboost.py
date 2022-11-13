datalist = [[0, 1, 3], [0, 3, 1], [1, 2, 2], [1, 1, 3], [1, 2, 3], [0, 1, 2], [7, 1, 2], [1, 1, 1], [1, 3, 1], [0 ,2 ,1]]
labellist = [-1] * 6 + [1, 1] + [-1, -1]
def createBoostTree(trainDataList, trainLabelList, treeNum = 4):
    

def loadData(filename):
    datalist, labellist = [], []
    f = open(filename)
    for line in f.readlines():
        line = line.strip().split(',')
        datalist.append([int(int(num) > 128) for num in line[1:]])
        labellist.append(1 if line[0] == '0' else -1)
    return datalist, labellist

import numpy as np
def createBoostTree(trainDataList, trainLabelList, treeNum = 50):
    '''
    创建提升树
    创建算法依据“8.1.2 AdaBoost算法” 算法8.1
    :param trainDataList:训练数据集
    :param trainLabelList: 训练测试集
    :param treeNum: 树的层数
    :return: 提升树
    '''
    #将数据和标签转化为数组形式
    trainDataArr = np.array(trainDataList)
    trainLabelArr = np.array(trainLabelList)
    #没增加一层数后，当前最终预测结果列表
    finallpredict = [0] * len(trainLabelArr)
    #获得训练集数量以及特征个数
    m, n = np.shape(trainDataArr)

    #依据算法8.1步骤（1）初始化D为1/N
    D = [1 / m] * m
    #初始化提升树列表，每个位置为一层
    tree = []
    #循环创建提升树
    for i in range(treeNum):
        #得到当前层的提升树
        curTree = createSingleBoostingTree(trainDataArr, trainLabelArr, D)
        #根据式8.2计算当前层的alpha
        alpha = 1/2 * np.log((1 - curTree['e']) / curTree['e'])
        #获得当前层的预测结果，用于下一步更新D
        Gx = curTree['Gx']
        #依据式8.4更新D
        #考虑到该式每次只更新D中的一个w，要循环进行更新知道所有w更新结束会很复杂（其实
        #不是时间上的复杂，只是让人感觉每次单独更新一个很累），所以该式以向量相乘的形式，
        #一个式子将所有w全部更新完。
        #该式需要线性代数基础，如果不太熟练建议补充相关知识，当然了，单独更新w也一点问题
        #没有
        #np.multiply(trainLabelArr, Gx)：exp中的y*Gm(x)，结果是一个行向量，内部为yi*Gm(xi)
        #np.exp(-1 * alpha * np.multiply(trainLabelArr, Gx))：上面求出来的行向量内部全体
        #成员再乘以-αm，然后取对数，和书上式子一样，只不过书上式子内是一个数，这里是一个向量
        #D是一个行向量，取代了式中的wmi，然后D求和为Zm
        #书中的式子最后得出来一个数w，所有数w组合形成新的D
        #这里是直接得到一个向量，向量内元素是所有的w
        #本质上结果是相同的
        D = np.multiply(D, np.exp(-1 * alpha * np.multiply(trainLabelArr, Gx))) / sum(D)
        #在当前层参数中增加alpha参数，预测的时候需要用到
        curTree['alpha'] = alpha
        #将当前层添加到提升树索引中。
        tree.append(curTree)

        #-----以下代码用来辅助，可以去掉---------------
        #根据8.6式将结果加上当前层乘以α，得到目前的最终输出预测
        finallpredict += alpha * Gx
        #计算当前最终预测输出与实际标签之间的误差
        error = sum([1 for i in range(len(trainDataList)) if np.sign(finallpredict[i]) != trainLabelArr[i]])
        #计算当前最终误差率
        finallError = error / len(trainDataList)
        #如果误差为0，提前退出即可，因为没有必要再计算算了
        if finallError == 0:    return tree
        #打印一些信息
        print('iter:%d:%d, sigle error:%.4f, finall error:%.4f'%(i, treeNum, curTree['e'], finallError ))
    #返回整个提升树
    return tree


def createSingleBoostingTree(trainDataArr, trainLabelArr, D):
    '''
    创建单层提升树
    :param trainDataArr:训练数据集数组
    :param trainLabelArr: 训练标签集数组
    :param D: 算法8.1中的D
    :return: 创建的单层提升树
    '''

    #获得样本数目及特征数量
    m, n = np.shape(trainDataArr)
    #单层树的字典，用于存放当前层提升树的参数
    #也可以认为该字典代表了一层提升树
    sigleBoostTree = {}
    #初始化分类误差率，分类误差率在算法8.1步骤（2）（b）有提到
    #误差率最高也只能100%，因此初始化为1
    sigleBoostTree['e'] = 1

    #对每一个特征进行遍历，寻找用于划分的最合适的特征
    for i in range(n):
        #因为特征已经经过二值化，只能为0和1，因此分切分时分为-0.5， 0.5， 1.5三挡进行切割
        for div in [-0.5, 0.5, 1.5]:
            #在单个特征内对正反例进行划分时，有两种情况：
            #可能是小于某值的为1，大于某值得为-1，也可能小于某值得是-1，反之为1
            #因此在寻找最佳提升树的同时对于两种情况也需要遍历运行
            #LisOne：Low is one：小于某值得是1
            #HisOne：High is one：大于某值得是1
            for rule in ['LisOne', 'HisOne']:
                #按照第i个特征，以值div进行切割，进行当前设置得到的预测和分类错误率
                Gx, e = calc_e_gx(trainDataArr, trainLabelArr, i, div, rule, D)
                #如果分类错误率e小于当前最小的e，那么将它作为最小的分类错误率保存
                if e < sigleBoostTree['e']:
                    sigleBoostTree['e'] = e
                    #同时也需要存储最优划分点、划分规则、预测结果、特征索引
                    #以便进行D更新和后续预测使用
                    sigleBoostTree['div'] = div
                    sigleBoostTree['rule'] = rule
                    sigleBoostTree['Gx'] = Gx
                    sigleBoostTree['feature'] = i
    #返回单层的提升树
    return sigleBoostTree

def calc_e_gx(trainDataArr, trainLabelArr, n, div, rule, D):
    '''
    计算分类错误率
    :param trainDataArr:训练数据集数字
    :param trainLabelArr: 训练标签集数组
    :param n: 要操作的特征
    :param div:划分点
    :param rule:正反例标签
    :param D:权值分布D
    :return:预测结果， 分类误差率
    '''
    #初始化分类误差率为0
    e = 0
    #将训练数据矩阵中特征为n的那一列单独剥出来做成数组。因为其他元素我们并不需要，
    #直接对庞大的训练集进行操作的话会很慢
    x = trainDataArr[:, n]
    #同样将标签也转换成数组格式，x和y的转换只是单纯为了提高运行速度
    #测试过相对直接操作而言性能提升很大
    y = trainLabelArr
    predict = []

    #依据小于和大于的标签依据实际情况会不同，在这里直接进行设置
    if rule == 'LisOne':    L = 1; H = -1
    else:                   L = -1; H = 1

    #遍历所有样本的特征m
    for i in range(trainDataArr.shape[0]):
        if x[i] < div:
            #如果小于划分点，则预测为L
            #如果设置小于div为1，那么L就是1，
            #如果设置小于div为-1，L就是-1
            predict.append(L)
            #如果预测错误，分类错误率要加上该分错的样本的权值（8.1式）
            if y[i] != L: e += D[i]
        elif x[i] >= div:
            #与上面思想一样
            predict.append(H)
            if y[i] != H: e += D[i]
    #返回预测结果和分类错误率e
    #预测结果其实是为了后面做准备的，在算法8.1第四步式8.4中exp内部有个Gx，要用在那个地方
    #以此来更新新的D
    return np.array(predict), e

def predict(x, div, rule, feature):
    if rule == 'LisOne':   
        L = 1; H = -1
    else:                   
        L = -1; H = 1

    #判断预测结果
    if x[feature] < div: 
        return L
    else:   
        return H
    
def model_test(testdatalist, testlabellist, tree):
    errorCnt = 0
    for i in range(len(testdatalist)):
        result = 0
        for curtree in tree:
            div = curtree['div']
            rule = curtree['rule']
            feature = curtree['feature']
            alpha = curtree['alpha']
            result += alpha * predict(testdatalist[i], div, rule, feature)
        if np.sign(result) != testlabellist[i]:
            errorCnt += 1
    return 1- errorCnt/len(testdatalist) 
   

import time 
if __name__ == '__main__':
    start = time.time()

    print('start read trainsSet')
    trainDataList, trainLabelList = loadData('D:\zenglinlin\data\mnist\mnist_train.csv')

    # 获取测试集
    print('start read testSet')
    testDataList, testLabelList = loadData('D:\zenglinlin\data\mnist\mnist_test.csv')

        #创建提升树
    print('start init train')
    tree = createBoostTree(trainDataList[:100], trainLabelList[:100], 1)

    #测试
    print('start to test')
    accuracy = model_test(testDataList[:1000], testLabelList[:1000], tree)
    print('the accuracy is:%d' % (accuracy * 100), '%')

    #结束时间
    end = time.time()
    print('time span:', end - start)
                    
                
            
     
# 8.1----------------------
# import numpy as np
# data_list = [[0, 1, 3], [0, 3, 1], [1, 2, 2], [1, 1, 3], [1, 2, 3], [0, 1, 2], [1, 1, 2],\
#     [1, 1, 1], [1, 3, 1], [0, 2, 1]]
# label_list = [-1] * 6 + [1, 1,] + [-1, -1]
# def createboost(data_list, label_list, tree_num):
#     data_array = np.array(data_list)
#     label_array = np.array(label_list)
#     final_prediction = [0] * len(label_array)
#     m, n = len(data_list), len(data_list[0])
#     w = np.array([1/m] * m)
#     tree = []

#     for num in range(tree_num):
#         cur_tree = create_single(data_array, label_array, w)
#         tree.append(cur_tree)
#         em = cur_tree['em']
#         alpha = 1/2 * np.log(1/em-1)
#         w = w* np.exp(alpha * np.multiply(label_array, cur_tree['gx']) * -1)/sum(w)
#         cur_tree['alpha'] = alpha
#         final_prediction += alpha * cur_tree['gx']
#         final = np.sign(final_prediction)
#         acc = sum(1*(final == label_array)) / m
#         print('iter: %d\tacc: %.4f\ttree_error: %.4f'%(num, acc, cur_tree['em']))
#         if acc > 0.9999:
#             break
#     print(tree)
    
# def create_single(data_array, label_array, w):
#     cur_tree = dict()
#     m, n = len(data_array), len(data_array[0])
#     min_error = float('inf')
#     for f in range(n):
#         for div in [-0.5, 0.5, 1.5, 2.5, 3.5]:
#             for rule in ['left', 'right']:
#                 error_rate, prediction_array = predict(data_array[:, f], label_array,div, w, rule)
#                 if error_rate < min_error:
#                     min_error = error_rate
#                     cur_tree['gx'] = prediction_array
#                     cur_tree['f'] = f
#                     cur_tree['div'] = div 
#                     cur_tree['rule'] = rule
#                     cur_tree['em'] = min_error

#     return cur_tree

# def predict(data, label, div, w, rule):

#     prediction = []
#     if rule == 'left': # left = -1
#         for d in data:
#             if d < div:
#                 prediction.append(-1)
#             else:
#                 prediction.append(1)
#     else:
#          for d in data:
#             if d >= div:
#                 prediction.append(-1)
#             else:
#                 prediction.append(1)
#     error_rate = 0
#     for i in range(len(label)):
#         if prediction[i] != label[i]:
#             error_rate += w[i]
#     return error_rate, np.array(prediction)

    
# if __name__ == '__main__':
#     createboost(data_list, label_list, 10)