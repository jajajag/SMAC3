import numpy as np
from smac.epm.hoag.abstract_hoag import AbstractHOAG
from sklearn.linear_model.logistic import _logistic_loss_and_grad, \
    _logistic_grad_hess, LogisticRegression
from scipy.sparse import linalg as splinalg
from smac.epm.hoag.abstract_hoag_functions import AbstractHOAGFunctions


class LRHOAG(AbstractHOAG, AbstractHOAGFunctions):

    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_valid: np.ndarray,
                 y_valid: np.ndarray,
                 maxiter_inner=10000,
                 epsilon_tol: float = 1e-3):
        """
                 bounds=None, disp=None, maxcor=10,
                 iprint=-1, maxls=20, tolerance_decrease='exponential',
                 verbose: bool = False,
        """
        AbstractHOAG.__init__(self)
        AbstractHOAGFunctions.__init__(self, X_train, y_train, X_valid, y_valid)
        self.epsilon_tol = epsilon_tol
        self.maxiter_inner = maxiter_inner

    def fit(self, lambdak):

        self.lambdak = lambdak
        # 调用lr计算模型X
        self.lr = LogisticRegression(C=self.lambdak[0], solver="liblinear",
                                fit_intercept=False)
        self.lr.fit(self.X_train, self.y_train)
        # 获得模型X和偏差
        self.model = self.lr.coef_.flatten()
        #self.intercept = lr.intercept_

        return self

    def predict_gradient(self):

        # 如果模型未经过训练
        if self.lambdak is None or self.model is None:
            raise ValueError("The model has not been fitted!")

        # 计算hessian矩阵
        fhs = self.h_hessian(self.model, alpha=self.lambdak)
        # B_op是hessian矩阵里，h对x的二阶导
        B_op = splinalg.LinearOperator(shape=(self.model.size, self.model.size),
                                       matvec=lambda z: fhs(z))

        g_func, g_grad = self.g_func_grad(self.model, alpha=self.lambdak)
        # 用来做梯度下降的参数
        Bxk = self.model.copy()
        tol_CG = self.epsilon_tol

        # (2) 从Ax=b，求解出x=A-1b来，使用splinalg库
        # B_op是二阶hession矩阵，g_grad是g的一阶梯度
        Bxk, success = splinalg.cg(B_op, g_grad, x0=Bxk, tol=tol_CG,
                                   maxiter=self.maxiter_inner)
        # 不成功，则报错
        if success != 0:
            raise ValueError('CG did not converge to the desired precision')

        # (3) 从pk得到lambdak的变化值(g的二阶梯度为0)
        grad_lambda = - self.h_crossed(self.model, alpha=self.lambdak).dot(Bxk)

        # 返回梯度信息
        return grad_lambda

    # 预测模型对测试集的概率
    def predict_proba(self, X):

        # 如果模型为None，则报错
        if self.model is None:
            raise ValueError("The model has not been fitted!")

        # return expit(np.dot(X, self.model) + self.intercept)
        return self.lr.predict_proba(X)

    # h和g的梯度等函数，调用sklearn里面的函数，视为私有函数
    def h_func_grad(self, x0, alpha=np.zeros(1), **kwargs):
        return _logistic_loss_and_grad(
            x0, self.X_train, self.y_train, 1 / alpha[0])

    def h_hessian(self, x0, alpha=np.zeros(1), **kwargs):
        return _logistic_grad_hess(
            x0, self.X_train, self.y_train, 1 / alpha[0])[1]

    def g_func_grad(self, x0, alpha=np.zeros(1), **kwargs):
        # alpha写死为0
        return _logistic_loss_and_grad(x0, self.X_valid, self.y_valid, 0)

    def h_crossed(self, x0, alpha=np.zeros(1), **kwargs):
        return 1 / alpha[0] * x0
