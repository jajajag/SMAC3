import scipy.sparse as sp
import numpy as np

def libsvm2sparse(lines):
    # 读入数据
    x_row = []
    x_col = []
    x_data = []
    y = []
    counter = 0
    
    for line in lines:
        # 如果是空行，则跳过
        line = line.strip()
        if line == '':
            continue
        # 将line分割开
        line = line.split()

        # 第一个元素为y值
        y.append(int(line[0]))

        # 遍历每个x的列
        for pos in line[1:]:
            pos = pos.split(':')
            # 行数
            x_row.append(counter)
            # 列数
            x_col.append(int(pos[0]) - 1)
            # 具体数值
            x_data.append(float(pos[1]))

        counter += 1

    # 返回稀疏数组x，包括数据，行列号，行数和最大列数
    x = sp.csr_matrix((x_data, (x_row, x_col)), shape=(counter, max(x_col) + 1))
    return x, np.array(y)

