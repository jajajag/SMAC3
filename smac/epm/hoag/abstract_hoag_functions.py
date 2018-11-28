import numpy as np


class AbstractHOAGFunctions(object):

    # 在HOAG Functions中存储数据集
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_valid: np.ndarray,
                 y_valid: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

    # 定义梯度矩阵的虚函数
    def h_func_grad(self, x0, **kwargs):
        raise NotImplementedError()

    def h_hessian(self, x0, **kwargs):
        raise NotImplementedError()

    def h_crossed(self, x0, **kwargs):
        raise NotImplementedError()

    def g_func_grad(self, x0, **kwargs):
        raise NotImplementedError()

    # 返回feature的个数
    def length(self):
        assert self.X_train.shape[1] == self.X_valid.shape[1]
        return self.X_train.shape[1]
