from smac.epm.gaussian_process.kernels import Cubic, Gaussian
import numpy as np
import matplotlib.pyplot as plt
import random


def plot_gp(x, g):

    mu, sigma = g.predict(x, return_std=True)
    plt.plot(x, mu, linewidth=3, label='Target')
    #axis.plot(bo.X.flatten(), bo.Y, 'D', markersize=8, label=u'Observations',
              #color='r')
    plt.plot(x, mu, '--', color='k', label='Prediction')

    plt.fill(np.concatenate([x, x[::-1]]),
              np.concatenate(
                  [mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='c', ec='None', label='95% confidence interval')

    #plt.set_xlim((-2, 10))
    #plt.set_ylim((None, None))
    #plt.set_ylabel('f(x)', fontdict={'size': 20})
    #plt.set_xlabel('x', fontdict={'size': 20})
    plt.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    plt.show()

our = True
points = 8
if our:
    from smac.epm.gaussian_process.gradient_gpr import GaussianProcessRegressor
else:
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.gpr import GaussianProcessRegressor

if __name__ == "__main__":
    # (1) 选取输入输出
    if our:
        g = GaussianProcessRegressor(kernel1=Cubic, kernel2=Gaussian)
    else:
        g = GaussianProcessRegressor(kernel=RBF())
    # 选取网格状的(X1, X2点对)
    X_test = np.arange(-8, 8, 0.25).reshape(-1, 1)
    #ind = np.random.choice(1024, 200)
    ind = random.sample(range(0, 64), points)
    X_train = X_test[ind]
    # 输出为y = X * sin(X)
    #y_train = np.sum(np.square(X_train), axis=1)
    y_train = (X_train * np.sin(X_train)).reshape(-1)
    # 梯度为y = sin(X) + X * cos(X)
    g_train = np.sin(X_train) + X_train * np.cos(X_train)

    # (2) 训练模型
    if our:
        g.fit(X_train, y_train, g_train)
    else:
        g.fit(X_train, y_train)

    # (3) 测试集输出
    # 绘制图像
    plot_gp(X_test, g)
