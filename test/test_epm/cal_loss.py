from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from smac.utils.libsvm2sparse import libsvm2sparse
import sys
#import matplotlib.pyplot as plt

if __name__ == "__main__":

    num_points = 1000

    # 读取输入文件
    with open(sys.argv[1], "r") as fp:
        df = fp.readlines()
    X, y = libsvm2sparse(df)
    """
    df = pd.read_csv(sys.argv[1])
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    """
    # 分割数据集
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=0.33,
                                                          random_state=1)

    # 计算所有点的loss值
    for i in range(num_points):
        #lr = LRHOAG(X_train, y_train, X_valid, y_valid)
        # 对于每个点，训练模型
        alpha = (i + 1) / num_points
        C = 1 / alpha
        lr = LogisticRegression(C=C, solver="liblinear", fit_intercept=False)
        lr.fit(X_train, y_train)
        #lr.fit([1 / alpha])
        #grad = lr.predict_gradient()
        #loss, grad = _logistic_loss_and_grad(lr.coef_.flatten(), X_valid,
        # y_valid, alpha)
        # 预测并计算loss
        y_pred = lr.predict_proba(X_valid)
        loss = log_loss(y_valid, y_pred)
        with open("output_logistic.txt", "a+") as fp:
            # 写入文件中
            fp.write(str(loss) + "\n")
            #fp.write(str(loss) + " " + str(grad) + "\n")
