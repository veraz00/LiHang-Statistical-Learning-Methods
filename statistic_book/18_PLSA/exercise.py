import numpy as np
def PLSA(WD, K):
    # WD is w-d rectangle
    # K is number of iterations 
    s = np.shape(WD)
    P_d = np.identity(s[1])/9    # P_d = p(di/dj)
    P_w_z = np.ones(shape = (s[0], K)) / s[0]  # P(wi/zk)
    P_z_d = np.ones(shape = (K, s[1]))/K  # p(zi/wd)
    for i in range(100):
        P_z_w_d = np.zeros(shape = (K, s[0], s[1]), dtype = np.float32)
        for k in range(K):
            for i in range(s[0]):
                for j in range(s[1]):
                    P_z_w_d[k, i, j] = P_w_z[i,k] *P_z_d[k,j] / np.matmul(P_w_z, P_z_d)[i, j]  # this is E step p(zk/wi, dj)
        
        # following is P step
        for i in range(s[0]):
            for k in range(K):
                Denominator = 0
                for m in range(s[0]):
                    for j in range(s[1]):
                        Denominator += WD[m, j] * P_z_w_d[k, m, j]
                molecule = 0
                for j in range(s[1]):
                    molecule += WD[i, j] * P_z_w_d[k, i, j]

                P_w_z[i, k] = molecule / Denominator  # updatep(wi/zk)
        for k in range(K):
            for j in range(s[1]):
                
                Denominator = sum(WD[:, j])
                molecule = 0.0
                for i in range(s[0]):
                    molecule += WD[i, j] * P_z_w_d[k, i, j]
                P_z_d[k, j] = molecule / Denominator  # update p(zk/dj)
    return P_z_w_d, P_w_z, P_z_d
if __name__ == '__main__':
    WD = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 1],
                   [0, 1, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 1],
                   [1, 0, 0, 0, 0, 1, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 0, 2, 0, 0, 1],
                   [1, 0, 1, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 1, 1, 0, 0, 0, 0]])
    P_z_w_d, P_w_z, P_z_d = PLSA(WD, K = 2)
    print(P_z_w_d, P_w_z, P_z_d)