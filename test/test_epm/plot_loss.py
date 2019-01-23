import matplotlib.pyplot as plt
import numpy as np

filename = "D:\\Downloads\\temp.txt"

if __name__ == "__main__":
    # 打开文件并处理字符串
    fp = open(filename, "r")
    lines = fp.readlines()
    lines = [i.strip() for i in lines]
    loss = []
    for line in lines:
        loss += line.split()
    # 获得loss列表和X坐标
    loss = [float(i) for i in loss]
    X = np.array(range(1000)) / 1000
    # 绘图
    plt.plot(X, loss)
    plt.show()