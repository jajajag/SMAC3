from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    # 绘制图像
    fig = plt.figure()
    ax = Axes3D(fig)

    X1 = np.arange(-4, 4, 0.25)
    X2 = np.arange(-4, 4, 0.25)
    X1, X2 = np.meshgrid(X1, X2)
    R = X1 ** 2 + X2 ** 2
    #R_grad =
    Z = R

    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap='rainbow')

    plt.show()
