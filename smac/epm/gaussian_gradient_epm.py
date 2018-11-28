import numpy as np

from smac.epm.base_epm import AbstractEPM
from smac.epm.gaussian_process.gradient_gpr import GaussianProcessRegressor
from smac.epm.gaussian_process.kernels import KernelBase, Cubic, Gaussian

__author__ = "jajajag"
__copyright__ = "Copyright 2018"
__license__ = "3-clause BSD"
__maintainer__ = "jajajag"
__email__ = "zzzsgsb@gmail.com"
__version__ = "0.0.1"


class GaussianGradientEPM(AbstractEPM):
    """Interface to the Gaussian EPM with gradient.

    Attributes
    ----------
    kernel1 : smac.utils.kernels.KernelBase
    kernel2 : smac.utils.kernels.KernelBase
    """

    def __init__(self,
                 kernel1: KernelBase = Gaussian,
                 kernel2: KernelBase = Cubic,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        kernel1 : smac.utils.kernels.KernelBase
            The first kernel in the Gaussian Process Regressor.
        kernel2 : smac.utils.kernels.KernelBase
            The second kernel in the Gaussian Process Regressor.
        """
        super().__init__(**kwargs)
        # 新建高斯回归器
        self.gpr = GaussianProcessRegressor(kernel1=kernel1, kernel2=kernel2)

    def _train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, ]
            The corresponding target values.

        Returns
        -------
        self
        """
        # 如果没给出梯度信息，则返回错误
        if 'gradient' not in kwargs:
            raise ValueError("Parameter gradient is not given.")

        # 训练高斯回归器
        self.gpr.fit(X, y.flatten(), kwargs['gradient'])

        """
        # 输出当前的预测的gp模型
        import numpy as np
        X = np.array(range(1000)) / 1000 + 0.001
        y = self.gpr.predict(X.reshape(-1, 1))
        print(y)
        """

        return self

    def _predict(self, X: np.ndarray):
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples,
                                   n_features (config + instance features)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        # 预测均值和标准差
        mean, std = self.gpr.predict(X, return_std=True)

        # 返回均值曲线和方差
        return mean.reshape((-1, 1)), np.square(std).reshape((-1, 1))
