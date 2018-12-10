from absl import app
from absl import flags
from smac.pssmac.facade import WorkerFacade, SchedulerFacade, ServerFacade
from smac.pssmac.tae.logistic_regression import LogisticRegression
from smac.utils.libsvm2sparse import libsvm2sparse
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 一些FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string("ps", "./smac/pssmac/ps_smac",
                    "Path to PS-Lite executable file.")
flags.DEFINE_string("data_dir", None, "Path to the data file.")
flags.DEFINE_string("node", None, "Type of the node(server/worker/scheduler)")
flags.DEFINE_string("temp_dir", "./tmp/", "Folder to save temporary files.")
# 默认1个server和1个worker
flags.DEFINE_integer("num_servers", 1, "Number of servers")
flags.DEFINE_integer("num_workers", 1, "Number of workers")
# scheduler和node对应的ip地址和端口号，需要指定
flags.DEFINE_string("scheduler_host", "127.0.0.1", "Host of the scheduler.")
flags.DEFINE_integer("scheduler_port", 8001, "Port of the scheduler.")
flags.DEFINE_string("node_host", "127.0.0.1", "Host of the server.")
flags.DEFINE_integer("node_port", 9600, "Port of the server.")
# 每个node对应的id
flags.DEFINE_integer("id", 0, "Id of current node.")
# 暂时还未将cutoff在worker和scheduler中实现
flags.DEFINE_integer("cutoff", 3600, "Total time of the whole smbo process.")
flags.DEFINE_integer("per_run_time_limit", 3600, "Time limit for one run.")
flags.DEFINE_integer("total_time_limit", 24 * 3600, "Total time limit.")


# 还可以将模型作为参数引入，mark下

def main(argv):
    """A fake-executable file in order to run the pssmac facade in a humanized
    way.

    Parameters
    ----------
    """
    # 大致检查一下输入的准确性
    ip_reg = re.compile("^(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|[1-9])\."
                        "(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)\."
                        "(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)\."
                        "(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)$")
    if not (ip_reg.match(FLAGS.scheduler_host) and ip_reg.match(
            FLAGS.node_host)):
        raise ValueError("Please check the host for scheduler and node.")
    if FLAGS.num_servers < 1 or FLAGS.num_workers < 1:
        raise ValueError("Please check the number of servers and workers.")
    if FLAGS.node is None:
        raise ValueError("Please specify the node type.")
    if FLAGS.scheduler_port < 0 or FLAGS.scheduler_port > 65535 or \
            FLAGS.node_port < 0 or FLAGS.node_port > 65535:
        raise ValueError("Please input a valid port number.")
    if FLAGS.cutoff < 0:
        raise ValueError("Please input a valid cutoff time.")

    # 处理ps-lite的参数
    scheduler_args = "role:SCHEDULER,hostname:'" + FLAGS.scheduler_host + \
                     "',port:" + str(FLAGS.scheduler_port) + ",id:'H'"
    ps_args = [FLAGS.ps,
               # 这个实际上是替换成node_args
               "-my_node", scheduler_args,
               "-scheduler", scheduler_args,
               "-num_servers", str(FLAGS.num_servers),
               "-num_workers", str(FLAGS.num_workers),
               "-log_dir", FLAGS.temp_dir + "pslog", "$@"]

    # 按类型进行分类调用，首先是scheduler
    if FLAGS.node.upper() == "SCHEDULER":
        # scheduler不需要改变ps_args的内容
        node = SchedulerFacade(ps_args)
        node.init(total_time_limit=FLAGS.total_time_limit)
    # 其次是server对应的分支
    elif FLAGS.node.upper() == "SERVER":
        # 提供给ps_args的参数
        server_args = "role:SERVER,hostname:'" + FLAGS.node_host + "'," \
                      + "port:" + str(FLAGS.node_port) + "," \
                      + "id:'S" + str(FLAGS.id) + "'"
        ps_args[2] = server_args
        # 先创建ServerFacade类
        node = ServerFacade(ps_args=ps_args)

        ta = LogisticRegression(np.array(0), np.array(0), np.array(0),
                                np.array(0))
        node.init(tae_runner=ta,
                  temp_folder=FLAGS.temp_dir,
                  our_work=None,
                  total_time_limit=FLAGS.total_time_limit)
    # 最后是worker对应的分支
    elif FLAGS.node.upper() == "WORKER":
        # worker剩下的参数
        worker_args = "role:WORKER,hostname:'" + FLAGS.node_host + "'," \
                      + "port:" + str(FLAGS.node_port) + "," \
                      + "id:'W" + str(FLAGS.id) + "'"
        ps_args[2] = worker_args
        # 先调用worker对应的facade
        node = WorkerFacade(ps_args=ps_args)

        # 读取输入文件
        if FLAGS.data_dir.endswith(".csv"):
            # 如果后缀名为csv则按csv读取
            df = pd.read_csv(FLAGS.data_dir)
            X = df[df.columns[:-1]].values
            y = df[df.columns[-1]].values
        else:
            # 否则按照libsvm读取
            with open(FLAGS.data_dir, "r") as fp:
                lines = fp.readlines()
            X, y = libsvm2sparse(lines)
            # 分割数据集
        # 分离数据集
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.33, random_state=1)
        ta = LogisticRegression(X_train, y_train, X_valid, y_valid)
        node.init(tae_runner=ta,
                  temp_folder=FLAGS.temp_dir,
                  worker_id=FLAGS.id,
                  per_run_time_limit=FLAGS.per_run_time_limit)
    else:
        raise ValueError("Please specify the node type.")

    # 调用node
    node.run()


if __name__ == "__main__":
    app.run(main)
