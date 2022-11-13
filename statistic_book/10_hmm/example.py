import numpy as np
import time 
# https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/01f6d6c9ebee258a61977137d3aadfcb5d997d04/HMM/HMM.py#L24


def trainParameter(fileName):
    statuDict = {'B': 0, 'M':1, 'E':2, 'S':3} 
    # B：词语的开头
    # M：一个词语的中间词
    # E：一个词语的结果
    # S：非词语，单个词
    #每个字只有四种状态，所以下方的各类初始化中大小的参数均为4
    #初始化PI的一维数组，因为对应四种状态，大小为4
    PI = np.zeros(4)  # statuDict = 'B', 'M', 'E', 'S'
    
    #初始化状态转移矩阵A，涉及到四种状态各自到四种状态的转移，因为大小为4x4
    A = np.zeros((4, 4))
    #初始化观测概率矩阵，分别为四种状态到每个字的发射概率
    #因为是中文分词，使用ord(汉字)即可找到其对应编码，这里用一个65536的空间来保证对于所有的汉字都能
    #找到对应的位置来存储
    B = np.zeros((4, 65536))
    # How many Chinese characters are stored in the computer
    # using the international Chinese character code? answer:No more than 65536.
    # one chinese charachter is 2 bytes; 1 byte is 8 bit; 2 byte is 16 bits; 2**16 = 65536
    #去读训练文本
    fr = open(fileName, encoding='utf-8')
    
    for line in fr.readlines():
        #---------------------训练集单行样例--------------------
                # 迈向  充满  希望  的  新  世纪  ——  一九九八年  新年  讲话  （  附  图片  １  张  ）  
# 中共中央  总书记  、  国家  主席  江  泽民  
# （  一九九七年  十二月  三十一日  ）
        #------------------------------------------------------
        #可以看到训练样本已经分词完毕，词语之间空格隔开，因此我们在生成统计时主要借助以下思路：
        # 1.先将句子按照空格隔开，例如例句中5个词语，隔开后变成一个长度为5的列表，每个元素为一个词语
        # 2.对每个词语长度进行判断：
        #       如果为1认为该词语是S，即单个字
        #       如果为2则第一个是B，表开头，第二个为E，表结束
        #       如果大于2，则第一个为B，最后一个为E，中间全部标为M，表中间词
        # 3.统计PI：该句第一个字的词性对应的PI中位置加1
        #           例如：PI = [0， 0， 0， 0]，当本行第一个字是B，即表示开头时，PI中B对应位置为0，
        #               则PI = [1， 0， 0， 0]，全部统计结束后，按照计数值再除以总数得到概率
        #   统计A：对状态链中位置t和t-1的状态进行统计，在矩阵中相应位置加1，全部结束后生成概率
        #   统计B：对于每个字的状态以及字内容，生成状态到字的发射计数，全部结束后生成概率
        #   注：可以看一下“10.1.1 隐马尔可夫模型的定义”一节中三个参数的定义，会有更清晰一点的认识
        #-------------------------------------------------------
        #对单行句子按空格进行切割
        curLine = line.strip().split() # [迈向, 充满, 希望, 的, 新, 世纪, ——, 一九九八年, 新年, 讲话, (,)]
        #对词性的标记放在该列表中
        wordLabel = []
        #对每一个单词进行遍历
        for i in range(len(curLine)):
            #如果长度为1，则直接将该字标记为S，即单个词
            if len(curLine[i]) == 1:
                label = 'S'
            else:
                #如果长度不为1，开头为B，最后为E，中间添加长度-2个M
                #如果长度刚好为2，长度-2=0也就不添加了，反之添加对应个数的M
                label = 'B' + 'M' * (len(curLine[i]) - 2) + 'E'

            #如果是单行开头第一个字，PI中对应位置加1,
            if i == 0: PI[statuDict[label[0]]] += 1

            #对于该单词中的每一个字，在生成的状态链中统计B
            for j in range(len(label)):
                #遍历状态链中每一个状态，并找到对应的中文汉字，在B中
                #对应位置加1
                B[statuDict[label[j]]][ord(curLine[i][j])] += 1

            #在整行的状态链中添加该单词的状态链
            #注意：extend表直接在原先元素的后方添加，
            #可以百度一下extend和append的区别
            wordLabel.extend(label)  # for A

        #单行所有单词都结束后，统计A信息
        #因为A涉及到前一个状态，因此需要等整条状态链都生成了才能开始统计
        for i in range(1, len(wordLabel)):
            #统计t时刻状态和t-1时刻状态的所有状态组合的出现次数
            A[statuDict[wordLabel[i - 1]]][statuDict[wordLabel[i]]] += 1
            
            
    sum = np.sum(PI)
    for i in range(len(PI)):
        if PI[i] == 0:
            PI[i] = 1e-8 
        else:
            PI[i] = np.log(PI[i]/sum) # change the probability into log
            
    for i in range(len(A)):
        sum = np.sum(A[i])
        for j in range(len(A[i])):
            if A[i][j] == 0:
                A[i][j] = 1e-8
            else:
                A[i][j] = np.log(A[i][j]/sum)
                
    for i in range(len(B)):
        sum = np.sum(B[i])
        for j in range(len(B[i])):
            if B[i][j] == 0:
                B[i][j] = 1e-8
            else:
                B[i][j] = np.log(B[i][j]/sum)
    
    return PI, A, B

def participate(artical, PI, A, B):
    retArtical= []
    for line in artical:  # line = ['去年１２月，我在广东深圳市出差，听说南山区工商分局为打工者建了个免费图书阅览室，这件新鲜事引起了我的兴趣。']
        delta = [[0 for i in range(4)]]
        for i in range(4):
            delta[0][i] = PI[i] + B[i][ord(line[0])] # PI[i] is not for first char, delta[0] is for 1st char
        psi = [[0 for i in range(4)] for i in range(len(line))]
        for t in range(1, len(line)):
            for i in range(4):
                tmpDelta = [0] * 4
                for j in range(4):
                    tmpDelta[j] = delta[t-1][j] + A[j][i]
                maxDelta = max(tmpDelta)
                maxDeltaIndex = tmpDelta.index(maxDelta)
                delta[t][i] = maxDelta + B[i][ord(line[t])] # for t th char in line, when get i label (i-0, 1, 2, 3), the proba
                psi[t][i] = maxDeltaIndex   #  for t th char in line, when get i label (i-0, 1, 2, 3), the index of previ label
        sequence = []
    sequence = []
    i_opt = delta[len(line) -1].index(max(delta[len(line) - 1]))  # the most post label for len(line)-1 th char
    sequence.append(i_opt)
    for t in range(len(line)-1, 0, -1):
        i_opt = psi[t][i_opt]
        sequence.append(i_opt)
    sequence.reverse()
    
        #开始对该行分词
    curLine = ''
    #遍历该行每一个字
    for i in range(len(line)):
        #在列表中放入该字
        curLine += line[i]
        #如果该字是3：S->单个词  或  2:E->结尾词 ，则在该字后面加上分隔符 |
        #此外如果改行的最后一个字了，也就不需要加 |
        if (sequence[i] == 3 or sequence[i] == 2) and i != (len(line) - 1):
            curLine += '|'
    #在返回列表中添加分词后的该行
        retArtical.append(curLine)
#返回分词后的文章
    return retArtical



def loadArticle(fileName):
    '''
    加载文章
    :param fileName:文件路径
    :return: 文章内容
    '''
    #初始化文章列表
    artical = []
    #打开文件
    fr = open(fileName, encoding='utf-8')
    #按行读取文件
    for line in fr.readlines():
        #读到的每行最后都有一个\n，使用strip将最后的回车符去掉
        line = line.strip()
        #将该行放入文章列表中
        artical.append(line)
    #将文章返回
    return artical


if __name__ == '__main__':
    # 开始时间
    start = time.time()

    #依据现有训练集统计PI、A、B
    PI, A, B = trainParameter('train.txt')

    #读取测试文章
    artical = loadArticle('test.txt')  # clean test.txt into [[,,], [,,], [,,]]

    #打印原文
    print('-------------------原文----------------------')
    for line in artical:
        print(line)

    #进行分词
    partiArtical = participate(artical, PI, A, B)

    #打印分词结果
    print('-------------------分词后----------------------')
    for line in partiArtical:
        print(line)

    #结束时间
    print('time span:', time.time() - start)