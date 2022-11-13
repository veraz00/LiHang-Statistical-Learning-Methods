import numpy as np


def gibbs_sampling_method(mean, cov, func, n_samples, m = 1000, random_state = 0):
    np.random.seed(random_state)
    n_features = len(mean)

    x0 = mean
    samples = []
    sum_ = 0
    for k in range(m+n_samples):
        x0[0] = np.random.multivariate_normal([x0[1]*cov[0][1]], np.diag([1-pow(cov[0][1], 2)]), 1)[0][0]  # here 
        x0[1] = np.random.multivariate_normal([x0[0] * cov[0][1]], np.diag([1 - pow(cov[0][1], 2)]), 1)[0][0]
        # mean=mean, cov=conv, size=N
    # np.random.multivariate_normal([x0[0] * cov[0][1]], np.diag([1 - pow(cov[0][1], 2)]), 1) -> [[1.63176551]]
              
        # 收集样本集合
        if k >= m:
            samples.append(x0.copy())
            sum_ += func(x0)

    return samples, sum_ / n_samples


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt


    def f(x):
        """目标求均值函数"""
        return x[0] + x[1]


    samples, avg = gibbs_sampling_method([0, 0], [[1, 0.5], [0.5, 1]], f, n_samples=10000)

    # print(samples)  # [[-2.0422584903207794, -2.5037388977869997], [-1.211915315832784, -1.4359343041313015], ...]
    print("样本目标函数均值:", avg)  # 0.0016714992469064399


    def draw_sample():
        """绘制样本概率密度函数的图"""
        X, Y = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
        Z = np.zeros((100, 100))
        for i, j in samples:
            Z[int(i // 0.1) + 50][int(j // 0.1) + 50] += 1

        fig = plt.figure()
        plt.imshow(Z, cmap="rainbow")
        plt.colorbar()
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
        plt.show()
