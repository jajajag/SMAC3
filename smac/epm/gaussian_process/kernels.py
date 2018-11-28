import numpy as np


class KernelBase(object):
    @staticmethod
    def base(X_train: np.ndarray,
             X_test: np.ndarray=None,
             metric: callable=lambda x, y: np.linalg.norm(x - y),
             fill: float=None,
             symmetric: bool=True):
        """Return the kernel or gradient of the input.

        Parameters
        ----------
        X_train : array, shape (n_samples_X_train, n_features)
            The train input values.

        X_test : array, shape (n_samples_X_test, n_features)
            The test input values.

        metric : lambda (optional, default=lambda x, y: np.linalg.norm(x - y))
            Apply the metric on each two sampling points of X.

        fill : float or int (optional, default = False)
            Fill the dianal with fill attribute.

        symmetric : bool (optional, default = True)
            Determine the matrix to be symmetric or skew_symmetric.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_X)
            Kernel k(X, Y)
        """
        # 检查输入是否是2-D
        if X_train.ndim != 2:
            raise ValueError("The train input is not a 2-D array.")
        N1, D = X_train.shape
        if X_test is not None:
            if X_test.ndim != 2 or X_test.shape[1] != D:
                raise ValueError("The test input is not a 2-D array.")
            N2, _ = X_test.shape

        # 检查距离函数是否可调用
        if not callable(metric):
            raise ValueError("The metric is not callable.")
        test_output = metric(np.zeros(D), np.zeros(D))
        # 检查输出函数的输出是0维还是1维
        if len(test_output.shape) == 0:
            if X_test is None:
                ret = np.zeros((N1, N1))
                # 如果是0维并且作为测试输出
            else:
                ret = np.zeros((N2, N1))
        elif len(test_output.shape) == 1:
            ret = np.zeros((N1, N1, D))
        else:
            raise ValueError(
                "The output of the metric is neither scalar or a 1-D array")

        if X_test is None:
            # 遍历所有点对
            for i in range(N1):
                for j in range(i + 1):
                    # 计算两点间距离
                    ret[i][j] = metric(X_train[i], X_train[j])
                    # 对称则填充相同元素，否则填充负值
                    ret[j][i] = ret[i][j] if symmetric else -ret[i][j]

            # 尝试填充对角线数值
            if fill is not None:
                try:
                    for i in range(N1):
                        ret[i][i] = fill
                except:
                    raise ValueError("The fill value cannot be assigned.")

        else:
            # 遍历所有点对
            for i in range(N2):
                for j in range(N1):
                    # 计算两点间距离
                    ret[i][j] = metric(X_test[i], X_train[j])

        return ret


class Gaussian(KernelBase):
    @staticmethod
    def kernel(X_train, X_test=None):
        # 高斯核函数: k = exp(-r^2)
        return KernelBase.base(X_train, X_test=X_test,
                               metric=lambda x, y: np.exp(
                                   -np.sum(np.square(x - y))), fill=1,
                               symmetric=True)

    def gradient(X_train, X_test=None):
        return KernelBase.base(X_train, X_test=X_test,
                               metric=lambda x, y: -2 * np.exp(
                                   -np.sum(np.square(x - y))) * (x - y), fill=0,
                               symmetric=False)


class Cubic(KernelBase):
    @staticmethod
    def kernel(X_train, X_test=None):
        # 立方函数：k = r ^ 3
        return KernelBase.base(X_train, X_test=X_test,
                               metric=lambda x, y: np.power(
                                   np.linalg.norm(x - y), 3), fill=0,
                               symmetric=True)

    @staticmethod
    def gradient(X_train, X_test=None):
        # 立方函数的梯度，设Xi = [Zi1, Zi2, ..., ZiD]，则
        # g = 3 * ((Zi1 - Zj1) ^ 2 + ... + (ZiD - ZjD) ^ 2) ^ 2 * (Zik - Zjk)
        # 对于任意两个点的某个维度
        # 创建一个数组存储结果，维度为(N, N, D)，其中N为采样点个数，D为features个数
        return KernelBase.base(X_train, X_test=X_test,
                               metric=lambda x, y: 3 * np.linalg.norm(x - y) * (
                                       x - y), fill=0,
                               symmetric=False)


class Linear(KernelBase):
    @staticmethod
    def kernel(X_train, X_test=None):
        # 线性函数：k = r
        return KernelBase.base(X_train, X_test=X_test,
                               metric=lambda x, y: np.linalg.norm(x - y),
                               fill=0,
                               symmetric=True)

    @staticmethod
    def gradient(X_train, X_test=None):
        # 立方函数的梯度，设Xi = [Zi1, Zi2, ..., ZiD]，则
        # g = 3 * ((Zi1 - Zj1) ^ 2 + ... + (ZiD - ZjD) ^ 2) ^ 2 * (Zik - Zjk)
        # 对于任意两个点的某个维度
        # 创建一个数组存储结果，维度为(N, N, D)，其中N为采样点个数，D为features个数
        return KernelBase.base(X_train, X_test=X_test,
                               metric=lambda x, y: (x - y) / (
                                       np.linalg.norm(x - y) + 1e-10),
                               fill=0,
                               symmetric=False)
