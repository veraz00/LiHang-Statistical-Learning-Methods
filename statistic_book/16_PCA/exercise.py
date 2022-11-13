import numpy as np
def PCA(X):
    s = np.shape(X)
    for i in range(s[0]):
        mean_i = np.mean(X[i, :])
        s_ii = np.sqrt(sum((X[i, :]-mean_i)**2)/(s[1] - 1))
        X[i,:] = (X[i, :] - mean_i) / s_ii
    X_T = X.T/np.sqrt(s[1]-1)
    conv_matrix = np.matmul(X_T.T, X_T)  # (2 * 2)
    print('conv_matrix: ', conv_matrix)
    U, char, V = np.linalg.svd(X_T)
    for i in range(s[0]):
        print(f'{i+1}th y has effects to x is:', char[i] * V[i, :], char[i]/np.sum(char))  # ??

if __name__ == '__main__':
    X = np.array([[2, 3, 3, 4, 5, 7],\
                [2, 4, 5, 5, 6, 8]])
    PCA(X)
