from ConfigSpace import Configuration, ConfigurationSpace, \
    UniformFloatHyperparameter, UnParametrizedHyperparameter,\
    CategoricalHyperparameter
from smac.pssmac.tae.abstract_tae import AbstractTAE
from sklearn.metrics import log_loss
import numpy as np
import sklearn


class LogisticRegression(AbstractTAE):
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
        AbstractTAE.__init__(self, X_train, y_train, X_valid, y_valid, metric)

    def set_config_space(self):
        self.cs = ConfigurationSpace()
        # 超参搜索空间，使用[1e-3, 1]
        # alpha = UniformFloatHyperparameter(name="alpha", lower=1e-3, upper=1,
        #                                   default_value=1, log=False)
        # 超参搜索空间，使用[1, 1000]
        C = UniformFloatHyperparameter(name="C", lower=1, upper=1000,
                                       default_value=1, log=False)
        """
        solver = UnParametrizedHyperparameter(
            name="solver", value="liblinear")
        # fit_intercept = UnParametrizedHyperparameter(
        #    name="fit_intercept", value=False)
        fit_intercept = CategoricalHyperparameter("fit_intercept", [False],
                                                  default_value=False)
        """
        self.cs.add_hyperparameters([C])#, solver, fit_intercept])

    def set_model(self):
        # 记得名字别冲突了
        self.model = sklearn.linear_model.LogisticRegression
