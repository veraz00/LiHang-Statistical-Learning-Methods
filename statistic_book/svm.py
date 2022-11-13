





# ----------------sklearn use support vector machine-----------------------------------
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
figure, ax1 = plt.subplots(nrows = 1, ncols = 1)
plt.figure(figsize=(10, 5))
def plot_svc_decision_function(model, ax = None, plot_support = True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0], xlim[1], 30) # (30,)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)  # (30. 30)
    print('P', P.shape)
    ax.contour(X, Y, P, colors = 'g', levels = [-1, 0, 1],
               alpha = 0.5, linestyles = ['--', '-', '--'])  # levels is name of level
    support_vector_indices = np.where(np.abs(P) <= 1+1e-15)[0]
    support_vectors_X = X[support_vector_indices]
    support_vectors_Y = Y[support_vector_indices]
    if plot_support:
        ax.scatter(support_vectors_X,
                   support_vectors_Y,
                   s = 300, linewidths=1, facecolor = 'none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()

def line_to_point_distance(p, q, r):
    def foo(t):
        x = t*(p-q) + q
        return np.linalg.norm(x, r)
    t0 = sci.optimize.minimize(foo, 0.1).x[0]
    return foo(t0)

    
X =np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, -1, -1])
clf = svm.LinearSVC()
clf.fit(X, Y)
print('clf', clf)
print('clf.coef_', clf.coef_)
print('clf.intercept_', clf.intercept_)

ax1.scatter(X[:, 0], X[:, 1], c = Y, s = 50, cmap = 'autumn')
plot_svc_decision_function(model = clf, ax = ax1)
decision_X = clf.decision_function(X)
support_vector_indice = np.where(np.abs(decision_X) <= 1 + 1e-15)[0]
support_vector = X[support_vector_indice]
print('support_vecotr', support_vector)

# print('distance', line_to_point_distance((0, -1 * clf.intercept_/ clf.coef_[1]), (-1* clf.intercept_/ clf.coef_[0], 0), v) for v in support_vector)
ax1.scatter(support_vector[:, 0], support_vector[:, 1],
            s = 300, lw = 1, facecolors = 'none')
plt.legend()
plt.show()