from ConfigSpace import Configuration
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np


class CustomizedTA(object):
    # 初始化函数
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_valid: np.ndarray,
                 y_valid: np.ndarray,
                 metric: callable = log_loss,
                 random_dataset: bool = False):
        """A customized ta function. Currently it only supports LR.

        Parameters
        ----------
        X_train : np.ndarray
        y_train : np.ndarray
        X_valid : np.ndarray
        y_valid : np.ndarray
        metric : callable, default_value = log_loss
            The metric function. It can be found in the sklearn package.
        random_dataset : bool, default_value = False
            Decide if we need to use randomized dataset. If set, we will patition
            the dataset into several datasets such that a single run of ta can
            be faster.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.metric = metric
        self.random_dataset = random_dataset

    def __call__(self, config: Configuration, seed: int = 12345):
        """The function that calculates the cost for a specific Configuration.

        Parameters
        ----------
        config : Configuration
            The incumbent or challenger in the intensifier.
        seed : int, default_value = 12345
            The seed used by some random generator.

        Returns
        -------
        loss : float
            The loss based on the model and Configuraiton we used.
        """
        # 根据alpha训练
        C = 1 / config._values['alpha']
        lr = LogisticRegression(C=C, solver="liblinear", fit_intercept=False)

        if self.random_dataset:
            X = np.concatenate((self.X_train, self.X_valid), axis=0)
            y = np.concatenate((self.y_train, self.y_valid), axis=0)
            # 使用训练集分割训练块和验证集，记得把训练集设为整个集合
            # 取得总大小
            size, _ = X.shape
            np.random.seed(0)
            rand_ind = np.random.permutation(range(size))
            # 将数据集分割成15份，每个训练集占2份，验证集用最后1/15
            pos = seed % 7
            start_train = int(2 * pos * size / 15)
            end_train = int(2 * (pos + 1) * size / 15)
            lr.fit(X[rand_ind[start_train: end_train]],
                   y[rand_ind[start_train: end_train]])
            start_valid = int(14 * size / 15)
            # 对loss进行预测
            y_pred = lr.predict_proba(X[rand_ind[start_valid:]])
            loss = self.metric(y[rand_ind[start_valid:]], y_pred)
        else:
            lr.fit(self.X_train, self.y_train)
            # 对loss进行预测
            y_pred = lr.predict_proba(self.X_valid)
            loss = self.metric(self.y_valid, y_pred)

        # ta函数只返回loss
        return loss
