from absl import app
from absl import flags
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from smac.utils.libsvm2sparse import libsvm2sparse
from smac.facade.oursmac_facade import ParallelSMBO
import datetime
import pandas as pd
import sys
import time

FLAGS = flags.FLAGS
# 默认不启用our_work和并行smbo
flags.DEFINE_string('our_work', None, 'Whether to use our code and the '
                                      'gradient file')
flags.DEFINE_integer('processes', 1, 'Number of processes')


# python3 test_oursmac.py filename [--our_work gradient_file_path]
# [--processes #processes]
def main(argv):
    if len(argv) < 2:
        raise ValueError("You should input filename.")
    if FLAGS.processes < 1 or FLAGS.processes > cpu_count():
        raise ValueError("Please input a proper number of processes.")
    # 读取输入文件
    if argv[1].endswith(".csv"):
        # 如果后缀名为csv则按csv读取
        df = pd.read_csv(sys.argv[1])
        X = df[df.columns[:-1]].values
        y = df[df.columns[-1]].values
    else:
        # 否则按照libsvm读取
        with open(argv[1], "r") as fp:
            lines = fp.readlines()
        X, y = libsvm2sparse(lines)
    # 分割数据集
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.33, random_state=1)

    processes = FLAGS.processes
    # 指定输入输出目录
    dirs = ["tmpfile/smac3-output_%s" % (
        datetime.datetime.fromtimestamp(time.time()).strftime(
            '%Y-%m-%d_%H:%M:%S_%f')) for _ in range(processes)]
    # 创建进程池
    pool = []
    for i in range(FLAGS.processes):
        cs = ConfigurationSpace()
        # 超参搜索空间，使用[1e-6, 1]
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=1e-3, upper=1,
            default_value=(i + 1) / processes, log=False)
        cs.add_hyperparameters([alpha])
        # 指定并行pSMAC目录和当前的输出目录
        pool.append(ParallelSMBO(X_train, y_train, X_valid, y_valid,
                                 dirs=dirs, smbo_id=i,
                                 cs=cs, our_work=FLAGS.our_work))
    for i in range(processes):
        pool[i].start()
    for i in range(processes):
        pool[i].join()


if __name__ == "__main__":
    app.run(main)
