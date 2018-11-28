import numpy as np

class AbstractHOAG(object):

    def __init__(self):
        # 将模型和超参初始化为None
        self.model = None
        self.lambdak = None

    def fit(self, lambda0):
        raise NotImplementedError()

    # 预测超参的梯度
    def predict_gradient(self):
        raise NotImplementedError()

    # 预测本超参对应的模型的概率
    def predict_proba(self, X):
        raise NotImplementedError()

    # 预测本超参对应的模型的值
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)