import pandas as pd
import sys
from libsvm2sparse import libsvm2sparse
import sklearn.model_selection
from hoag import LogisticRegressionCV


# 调用格式为python3 output_values.py 文件名 lambda值
# 调用格式为python3 output_values.py real-sim 0
# 返回值分别是g函数，模型X，lambda和g函数的导数pk
# 只适用于离散数据
if __name__ == '__main__':
    # 从文件名读取数据
    with open(sys.argv[1], 'r') as fp:
        lines = fp.readlines()

    # 分类训练集和测试集
    X, y = libsvm2sparse(lines)
    Xt, Xh, yt, yh = sklearn.model_selection.train_test_split(
        X, y, random_state=1, stratify=y)

    if len(sys.argv) > 2:
        lambdak = float(sys.argv[2])
    else:
        lambdak = .0

    #print(lambdak)

    clf = LogisticRegressionCV(max_iter=1, verbose=False, alpha0=lambdak)
    g_func, x, _lambdak, grad_lambda = clf.fit(Xt, yt, Xh, yh)

    with open('output.txt', 'w') as fp:
        fp.write(str(g_func) + '\n')
        fp.write(' '.join([str(i) for i in x]) + '\n')
        fp.write(str(lambdak) + '\n')
        fp.write(str(grad_lambda) + '\n')
        #fp.write(str(_lambdak) + '\n')
