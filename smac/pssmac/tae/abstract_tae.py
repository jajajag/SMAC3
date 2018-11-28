from ConfigSpace import Configuration, ConfigurationSpace, \
    UniformFloatHyperparameter, UnParametrizedHyperparameter
from sklearn.metrics import log_loss
import numpy as np


class AbstractTAE(object):
    # 初始化函数
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_valid: np.ndarray,
                 y_valid: np.ndarray,
                 metric: callable = log_loss):
        """A customized ta function. Currently it only supports LR.

        Parameters
        ----------
        X_train : np.ndarray
        y_train : np.ndarray
        X_valid : np.ndarray
        y_valid : np.ndarray
        metric : callable, default_value = log_loss
            The metric function. It can be found in the sklearn package.
        """
        # 这几个是在外面赋值
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.metric = metric
        # 这两个是模型相关的，需要覆写对应函数
        self.set_config_space()
        self.set_model()
        # Fxxk，千万不要又定义又返回

    def __call__(self, config: Configuration, seed: int = 12345) -> float:
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
        # 将保存了超参的数组直接传入模型中
        cls = self.model(**config._values)
        cls.fit(self.X_train, self.y_train)
        # 对loss进行预测
        try:
            y_pred = cls.predict_proba(self.X_valid)
            loss = self.metric(self.y_valid, y_pred)
        except:
            # 省事了，如果是预测出label的metric，这么调用
            # 不服的话，请覆写这部分函数
            y_pred = cls.predict(self.X_valid)
            loss = self.metric(self.y_valid, y_pred)

        # ta函数只返回loss
        return loss

    def set_config_space(self) -> None:
        """Method to set the ConfigurationSpace. It should contain all the
        possible hyperparameters in the space. The default values should also be
        given.

        Returns
        -------
        """
        raise NotImplementedError

    def get_config_space(self) -> ConfigurationSpace:
        """Return the configuration space.

        Returns
        -------
        cs : ConfigurationSpace
            The configuration related to the ta.
        """
        return self.cs

    def set_model(self) -> None:
        """Method to set the model. The model should be the class name but not
        the class object. It should be overrided.

        Returns
        -------
        """
        raise NotImplementedError

    def get_model(self) -> callable:
        """Return the model.

        Returns
        -------
        model : callable
            Return the model name.
        """
        return self.model
