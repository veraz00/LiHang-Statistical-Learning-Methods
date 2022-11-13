import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import time 

def load_data(file):
    '''
    INPUT:
    file - (str) 数据文件的路径
    
    OUTPUT:
    org_topics - (list) 原始话题标签列表 -- k
    text - (list) 文本列表-- t
    words - (list) 单词列表--n
    
    '''
    df = pd.read_csv(file)  # every row is a text 
    org_topics = df['category'].unique().tolist()
    df.drop('category', axis = 1, inplace = True)
    n = df.shape[0] # number of text
    text, words = [], []
    for i in df['text'].values:
        # str.maketrans('a', 'A', '0123456789') 
        # translate 'a' into 'A ' and any char in '01239' is get rid of
        t = i.translate(str.maketrans('', '',string.punctuation))
        t = [j for j in t.split() if j not in stopwords.words('english')]
        t = [j for j in t if len(j) >3]
        text.append(t)
        words.extend(set(t))
    words = list(set(words))
    return org_topics, text, words  # topic, cleaned text, unique words

def frequency_counter(text, words):
    '''
    INPUT:
    text - (list) 文本列表
    words - (list) 单词列表
    
    OUTPUT:
    X - (array) 单词-文本矩阵
    
    '''
    X = np.zeros((len(words), len(text)))
    for i in range(len(text)):
        t = text[i]
        for w in t:
            ind = words.index(w)
            X[ind][i] += 1
    return X

def do_lsa(X, k, words):
    '''
    INPUT:
    X - (array) 单词-文本矩阵
    k - (int) 设定的话题数
    words - (list) 单词列表
    
    OUTPUT:
    topics - (list) 生成的话题列表
    
    '''
    w, v = np.linalg.eig(np.matmul(X.T, X))
     #计算Sx的特征值和特征向量，其中Sx=X.T*X，Sx的特征值w即为X的奇异值分解的奇异值，v即为对应的奇异向量
    sort_inds = np.argsort(w)[::-1] 
    V_T = []  #用来保存矩阵V的转置
    for ind in sort_inds:
        V_T.append(v[ind]/np.linalg.norm(v[ind]))  #将降序排列后各特征值对应的特征向量单位化后保存到V_T中
    V_T = np.array(V_T)  #将V_T转换为数组，方便之后的操作
    Sigma = np.diag(np.sqrt(w))  #将特征值数组w转换为对角矩阵，即得到SVD分解中的Sigma
    U = np.zeros((len(words), k))  #用来保存SVD分解中的矩阵U  # U(x, k) * sigma * v (k,t)

    for i in range(k):
        ui = np.matmul(X, V_T.T[:, i]) / Sigma[i][i]  #计算矩阵U的第i个列向量
        U[:, i] = ui  #保存到矩阵U中

    topics = []  #用来保存k个话题
    for i in range(k):
        inds = np.argsort(U[:, i])[::-1]  #U的每个列向量表示一个话题向量，话题向量的长度为m，其中每个值占向量值之和的比重表示对应单词在当前话题中所占的比重，这里对第i个话题向量的值降序排列后取出对应的索引值
        topic = []  #用来保存第i个话题
        for j in range(10):
            topic.append(words[inds[j]])  #根据索引inds取出当前话题中比重最大的10个单词作为第i个话题
        topics.append(' '.join(topic))  #保存话题i
    return topics



if __name__ == '__main__':
    org_topics, text, words = load_data('data.csv')
    print('Origin Topcis: ')
    print(org_topics)
    start = time.time()
    X = frequency_counter(text, words)
    k = 5 # the number of topics are 5
    topics = do_lsa(X, k, words)
    print('Generate topics:')
    for i in range(k):
        print('Topic {}: {}'.format(i+1, topics[i]))
    end = time.time()
    print('Time: ', end-start)