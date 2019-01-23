from smac.configspace import Configuration, ConfigurationSpace
from smac.configspace.util import convert_configurations_to_array
from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory
from smac.tae.execute_ta_run import StatusType
import numpy as np
import subprocess
import typing


class AbstractPS(object):
    def __init__(self,
                 ps_args: typing.List[str],
                 cs: ConfigurationSpace,
                 aggregate_func: callable = average_cost) -> None:
        """Initialize AbstractPS.

        Parameters
        ----------
        ps_args : typing.List[str]
            List of strings that are used to open a PS-Lite
            server/worker/scheduler.
        cs : ConfigurationSpace, default_value = None
            ConfigurationSpace of the hyperparameters.
        aggregate_func : callable
            Aggregate function for RunHistory.
        """
        self.ps = subprocess.Popen(ps_args, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
        self.cs = cs
        self.aggregate_func = aggregate_func

    def push(self, **kwargs) -> None:
        """Push the string parsed from push_parser to the other side.

        Parameters
        ----------
        kwargs : typing.Dict
            Parameters of push_parser function for server/worker.

        Returns
        -------
        """
        # 按行读写，防止缓冲区爆炸
        time_left, lines = self.push_parser(**kwargs)
        # 每行结尾加上\n，表示换行
        lines = [(line + "\n").encode("ascii") for line in lines]

        # 计算超参的个数
        num_config = len(self.cs.get_hyperparameters())
        # 首先，输出总行数以及每个Configuration中超参的个数
        self.ps.stdin.write(
            (str(len(lines)) + " " + str(num_config) + " " + str(
                time_left) + "\n").encode("ascii"))
        self.ps.stdin.flush()

        # push函数，将内容parse后推送给server/worker端
        for line in lines:
            # 按行输出到pipe
            self.ps.stdin.write(line)
            self.ps.stdin.flush()

    def pull(self):
        """Pull and parse the data from the other node.

        Returns
        -------
        """
        line = ""
        # 略去空行
        while line == "":
            line = self.ps.stdout.readline().decode("ascii").strip()
        line = line.split()
        # 读入runhistory的个数和num_config的个数
        num_runhistory, num_config, time_left = int(float(line[0])), int(
            float(line[1])), float(line[2])
        # 最新修改，加上剩余时间

        # 空列表，存储所有的ConfigHistory
        ret_list = []
        while len(ret_list) < num_runhistory:
            line = self.ps.stdout.readline().decode("ascii").strip()
            # 如果读取到空行，则跳过
            if line == "":
                continue
            # 否则加入ret_list豪华午餐
            ret_list.append(line)

        # pull函数，返回parse后的拉取的数据
        return self.pull_parser(time_left, ret_list)

    def push_parser(self, **kwargs) -> typing.Tuple[float, typing.List[str]]:
        """Parse the data to a string that can be passed from Python to C++
            program.

        Parameters
        ----------
        kwargs : typing.Dict
            Parameters of push_parser function for server/worker.

        Returns
        -------
        time_left : float
            Return the time left for the procedure.
        return : typing.List[str]
            Return a list of string per config/runhistory on each line.
        """
        # 定义两个个虚函数，用来处理具体的通讯细节
        raise NotImplementedError

    def pull_parser(self, time_left: float, data: typing.List[str]):
        """Parse List[str] data to their original forms.

        Parameters
        ----------
        time_left : float
            The time left for the smbo.
        data : typing.List[str]
            Data passed from the node.

        Returns
        -------
        """
        raise NotImplementedError


class ConfigHistory(object):
    def __init__(self,
                 config: Configuration,
                 cs: ConfigurationSpace,
                 runhistory: RunHistory = None,
                 aggregate_func: callable = average_cost):
        """A tuple for config and its runhistory.

        Parameters
        ----------
        config : Configuration
            The configuration. It may be incumbent or one of the challengers.
        cs : ConfigurationSpace
            ConfigSpace of the model.
        runhistory : RunHistory, defualt_value = None
            The runhistory of the config. It can be either empty or conatins
            more configurations than the given one. If runhistory is not givem,
            I will set it to a empty RunHistory object.
        aggregate_func : callable, defualt_value = average_cost
            The aggregate function.
        """
        self.config = config
        self.cs = cs
        self.aggregate_func = aggregate_func
        # 初始化runhistory
        self.runhistory = RunHistory(
            aggregate_func=aggregate_func) if runhistory is None else runhistory

    def to_str(self) -> str:
        """Convert the ConfigHistory object to a string.

        Returns
        -------
        return : str
            A str contains the Configuration and its related runhistories.
            For example, "0.8(config) 1(#runhistory) 0.6 1.2 1234"
        """
        # 取得config对应的runhistory
        runhistory = self.runhistory.get_history_for_config(self.config)

        # 将config转化为ndarray
        config_list = [str(param) for param in
                       convert_configurations_to_array([self.config])[0]]
        # 将runhistory转化为字符串list，每个元素为"$cost $time $seed"形式
        runhistory_list = [" ".join([str(item) for item in history]) for history
                           in runhistory]

        # 返回由config，runhistory数量和各个runhistory字符串组成的list
        return " ".join(config_list + [str(len(runhistory_list))] +
                        runhistory_list)

    @staticmethod
    def read_str(data: str,
                 cs: ConfigurationSpace,
                 aggregate_func: callable = average_cost):
        """Read a string line and transform it to a ConfigHistory. The input
        should be valid. For example, "0.8(config) 1(#runhistory) 0.6 1.2 1234"

        Parameters
        ----------
        data : str
            A list of strings containing config and runhistory info.
        cs : ConfigurationSpace
            The ConfigurationSpace.
        aggregate_func : callable, default = average_cost
            The aggregate function.

        Returns
        -------
        Return : ConfigHistory
            Return a ConfigHistory.
        """
        # 首先将这行分开，读入Configuration
        line = data.split()
        # 用ConfigSpace计算超参的个数
        num_config = len(cs.get_hyperparameters())
        config = Configuration(cs, vector=np.array(
            [float(param) for param in line[:num_config]]))

        # 初始化参数，每个config对应一个runhistory
        runhistory = RunHistory(aggregate_func=aggregate_func)
        # 读取runhistory的数量
        num_runhistory = int(float(line[num_config]))
        counter = num_config + 1
        # 之后，读取每三对的数作为runhistory
        for i in range(num_runhistory):
            cost = float(line[counter])
            time = float(line[counter + 1])
            seed = int(float(line[counter + 2]))
            counter += 3
            # 添加到runhistory
            runhistory.add(config, cost, time, StatusType.SUCCESS, seed=seed)

        # 返回本个runhistory
        config_history = ConfigHistory(config, cs, runhistory=runhistory,
                                       aggregate_func=aggregate_func)
        return config_history

    def get_config(self) -> Configuration:
        """Return the Configuration of the object.

        Returns
        -------
        config : Configuration
        """
        return self.config

    def get_runhistory(self) -> RunHistory:
        """Return the RunHistory of the object.

        Returns
        -------
        runhistory : RunHistory
        """
        return self.runhistory
