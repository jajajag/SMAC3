from smac.epm.gaussian_process.kernels import Cubic, Gaussian
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

our = True
points = 32
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
    X1 = np.arange(-4, 4, 0.25)
    X2 = np.arange(-4, 4, 0.25)
    X1, X2 = np.meshgrid(X1, X2)
    # 重排的到测试集的点对
    X_test = np.array([X1.reshape(-1), X2.reshape(-1)]).T
    # 随机选择100个点作为采样点
    #ind = np.random.choice(1024, 200)
    ind = random.sample(range(0, 1024), points)
    X_train = X_test[ind]
    # 输出为y = X1 ^2 + X2 ^ 2
    y_train = np.sum(np.square(X_train), axis=1)
    # 梯度为y = [2 * X1, 2 * X2]
    g_train = 2 * X_train

    # (2) 训练模型
    if our:
        g.fit(X_train, y_train, g_train)
    else:
        g.fit(X_train, y_train)

    # (3) 测试集输出
    # 绘制图像
    fig = plt.figure()
    ax = Axes3D(fig)
    y_test = g.predict(X_test)
    y = y_test.reshape(32, 32)

    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    ax.plot_surface(X1, X2, y, rstride=1, cstride=1, cmap='rainbow')

    plt.show()
