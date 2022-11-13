import numpy as np

def pagerank(r0, max_step = 100):
    M = np.array([[0,1/2,1,0],[1/3,0,0,1/2],[1/3,0,0,1/2],[1/3,1/2,0,0]])  # transform, colum to row
    rt = r0
    for i in range(max_step):
        rt = np.dot(M, rt)
    return rt

if __name__ == '__main__':
    results = []
    test = [[1/4,1/4,1/4,1/4],[1/2,0,1/2,0],[1/3,1/3,1/3,0],[1/3,1/2,0,1/6],[1/3,1/4,1/6,1/4]]  # 5 different intial valuye
    for i in range(len(test)):
        result = pagerank(test[i])
        results.append(result)
    print(results)



