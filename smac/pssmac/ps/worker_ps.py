from multiprocessing import Process
from smac.configspace import Configuration, ConfigurationSpace
from smac.intensification.intensification import Intensifier
from smac.optimizer.objective import average_cost
from smac.pssmac.ps.abstract_ps import AbstractPS, ConfigHistory
from smac.runhistory.runhistory import RunHistory
import typing
import time


class Worker(AbstractPS, Process):
    def __init__(self,
                 ps_args: typing.List[str]):
        """Initialize worker.

        Parameters
        ----------
        ps_args : typing.List[str]
            List of strings that are used to open a PS-Lite
            server/worker/scheduler.
        """
        # 显式调用初始化self.ps
        AbstractPS.__init__(self, ps_args)
        # 非常重要，不然不能调用多进程类
        Process.__init__(self)
        self.intensifier = None
        self.worker_id = 0

    def init(self,
             cs: ConfigurationSpace,
             aggregate_func: callable = average_cost,
             **kwargs):
        """Initialize the worker ps.

        Parameters
        ----------
        cs : ConfigurationSpace, default_value = None
            ConfigurationSpace of the hyperparameters.
        aggregate_func : callable
            Aggregate function for RunHistory.
        kwargs["intensifier"] : Intensifier
            The intensifier of the SMBO algorithm. It runs ta function and find
            the best n losses.
        kwargs["worker_id"] : int, default_value = 0
            Worker_id, which is used as the seed of random number generator.

        Returns
        -------

        """
        AbstractPS.init(self, cs, aggregate_func=aggregate_func)
        if "intensifier" not in kwargs:
            raise AttributeError("Missing intensifier in kwargs!")
        self.intensifier = kwargs["intensifier"]
        # worker_id 初始值为0
        if "worker_id" in kwargs:
            self.worker_id = kwargs["worker_id"]

    # 处理中间过程的函数
    def run(self) -> None:
        """The main loop for the worker.

        Returns
        -------
        None
        """
        # 首次处理，产生随机的config
        self.initial_run()
        print("Initialized from worker ", self.worker_id)
        # 初始化剩余时间
        time_left = float('inf')
        # worker的主循环
        while time_left > 0:
            # 拉取server传来的参数
            incumbent, runhistory, challengers, time_left = self.pull()
            print("Pulled from worker ", self.worker_id)
            wall_time = time.time()

            # 使用intersifier进行计算
            new_incumbent, new_runhistory = self.solver(incumbent, runhistory,
                                                        challengers, time_left)
            # 推送计算结果给server
            self.push(incumbent=new_incumbent, runhistory=new_runhistory,
                      time_left=time_left)
            print("Pushed from worker ", self.worker_id)
            # 计算新剩余的时间
            time_left -= time.time() - wall_time

    def solver(self, incumbent: Configuration, runhistory: RunHistory,
               challengers: typing.List[Configuration], time_left: float) -> \
            typing.Tuple[Configuration, RunHistory]:
        """Find the next incumbent from the challengers and add the runhistory
        of every ta runs.

        Parameters
        ----------
        incumbent : Configuration
            Current best configuration.
        runhistory : RunHistory
            The runhistory that contains the incumbent.
        challengers : typing.List[Configuration]
            Challengers of the SMBO.

        Returns
        -------
        new_incumbent : Configuration
        runhistory : RunHistory
        """
        self.intensifier.tae_runner.runhistory = runhistory
        # 用intensifier寻找新的最优值，并运行ta
        new_incumbent, _ = self.intensifier.intensify(
            challengers=challengers,
            incumbent=incumbent,
            run_history=runhistory,
            aggregate_func=self.aggregate_func,
            # 加入时间条件，防止运行过久
            time_bound=max(self.intensifier._min_time, time_left))

        return new_incumbent, runhistory

    def initial_run(self):
        """Run the initial design and then push the results to the server.

        Returns
        -------
        Return : None
            Only write the results to the worker.
        """
        # 第一次运行，先随机一个Configuration，然后运行ta并存入runhistory
        # 用worker_id产生随机数
        seed = self.cs.seed(self.worker_id)
        incumbent = self.cs.sample_configuration()
        incumbent.origin = 'Random initial design.'
        # 创建一个空的runhistory
        runhistory = RunHistory(aggregate_func=self.aggregate_func)
        # 用空runhistory替换tae_runner中的runhistory
        self.intensifier.tae_runner.runhistory = runhistory

        # 先创建一个随机的instance
        rand_inst = self.intensifier.rs.choice(list(self.intensifier.instances))
        self.intensifier.tae_runner.start(incumbent, instance=rand_inst)

        self.push(incumbent=incumbent, runhistory=runhistory, time_left=3600)

    def push_parser(self, **kwargs) -> typing.Tuple[float, typing.List[str]]:
        """Parse the incumbent, used challengers and related runhistory to a
        string.

        Parameters
        ----------
        kwargs["incumbent"] : Configuration
            The current best Configuration.
        kwargs["runhistory"] : Runhistory
            The Runhistory of the whole process of intensifier.
        kwargs["time_left"] : float
            The time left for the smbo(useless here).

        Returns
        -------
        time_left : float
            Time left for the SMBO process.
        return : typing.List[str]
            Return the string to be passed to the PS-Lite worker.
        """
        if not (
                "incumbent" in kwargs and "runhistory" in kwargs and "time_left" in kwargs):
            raise KeyError("At least one of the arguments: [incumbent, \
                                runhistory, time_left] are missing.")

        # 取得runhistory中所有的configs
        configs = kwargs["runhistory"].get_all_configs()
        incumbent = ConfigHistory(kwargs["incumbent"], self.cs,
                                  runhistory=kwargs["runhistory"])
        # challengers中不应包括incumbent，configspace中定义了不等号
        challengers = [ConfigHistory(challenger, self.cs,
                                     runhistory=kwargs["runhistory"]).to_str()
                       for challenger in configs if
                       challenger != incumbent.get_config()]
        time_left = kwargs["time_left"]

        # 返回incumbent和challengers对应的str的list
        return time_left, [incumbent.to_str()] + challengers

    def pull_parser(self, time_left: float, data: typing.List[str]) -> \
            typing.Tuple[
                Configuration, RunHistory, typing.List[Configuration], float]:
        """Parse incumbent. its runhistory and the challengers

        Parameters
        ----------
        time_left:
            Time left for the smbo procedure.
        data : typing.List[str]
            List of strings that contain incumbent, challengers and runhistory.

        Returns
        -------
        incumbent : Configuration
            Current best configuration.
        runhistory : RunHistory
            Runhistory of the incumbent configuration.
        challengers : typing.List[Configuration]
            List that holds all the challengers.
        time_left : float
            Time left for the smbo procedure.
        """
        # 获得第一个incumbent的ConfiguHistory，需要both Config和Runhistory
        incumbent = ConfigHistory.read_str(data[0], self.cs,
                                           aggregate_func=self.aggregate_func)
        # 读取剩下所有行的Configuration，作为challengers
        challengers = [ConfigHistory.read_str(
            line, self.cs, aggregate_func=self.aggregate_func).get_config()
                       for line in data[1:]]

        # 依次返回incumbent，runhistory和challengers
        return incumbent.get_config(), incumbent.get_runhistory(), \
               challengers, time_left
