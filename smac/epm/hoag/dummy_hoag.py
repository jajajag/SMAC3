import numpy as np
from smac.epm.hoag.abstract_hoag import AbstractHOAG
import math


class DummyHOAG(AbstractHOAG):

    def __init__(self,
                 lower: float,
                 upper: float,
                 gradient: np.ndarray,
                 ):
        # 搜索空间的上下限
        self.lower = lower
        self.upper = upper
        self.gradient = np.asarray(gradient)
        self.lambdak = None
        self.eta = 1e-6

    def fit(self, lambda0: np.ndarray):
        self.lambdak = lambda0

    # 预测超参的梯度
    def predict_gradient(self):
        if self.lambdak is None:
            raise ValueError("The model has not been fitted!")

        #if self.lambdak[0] < self.lower or self.lambdak[0] > self.upper:
        #    raise ValueError("The hyperparameter is out of bound.")
        if self.lambdak[0] < self.lower:
            return self.gradient[0]

        if self.lambdak[0] > self.upper:
            return self.gradient[-1]

        # 获取列表长度
        n = self.gradient.shape[0]
        step_size = (self.upper - self.lower) / n

        # 取下界到gradient数组
        return self.gradient[math.floor(max(self.lambdak[0] - self.lower - self.eta, 0) / step_size)]


    def predict_proba(self, X):
        pass

    def predict(self, X):
        pass
