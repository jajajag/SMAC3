import pandas as pd
import sys
from libsvm2sparse import libsvm2sparse
import sklearn.model_selection
from hoag import LogisticRegressionCV

if __name__ == '__main__':
    # 从文件名读取数据
    with open(sys.argv[1], 'r') as fp:
        lines = fp.readlines()

    # 分类训练集和测试集
    X, y = libsvm2sparse(lines)
    Xt, Xh, yt, yh = sklearn.model_selection.train_test_split(
        X, y, random_state=1, stratify=y)

    clf = LogisticRegressionCV(max_iter=50, verbose=True)
    clf.fit(Xt, yt, Xh, yh)
