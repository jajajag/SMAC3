from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from multiprocessing import Process
from smac.epm.gaussian_gradient_epm import GaussianGradientEPM
from smac.epm.hoag import AbstractHOAG, DummyHOAG
from smac.facade.smac_facade import SMAC
from smac.intensification.intensification import Intensifier
from smac.optimizer.objective import average_cost
from smac.pssmac import Worker, Server
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.tae.execute_ta_customized import CustomizedTA
from smac.tae.execute_ta_run import StatusType
from smac.utils.io.traj_logging import TrajLogger
import datetime
import numpy as np
import subprocess
import time
import typing


class PS_SMAC(Process):
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_valid: np.ndarray,
                 y_valid: np.ndarray,
                 cs: ConfigurationSpace,
                 temp_folder: str,
                 scheduler_args: typing.List[str],
                 server_args: typing.List[str],
                 worker_args: typing.List[typing.List[str]],
                 our_work: str = None,
                 cutoff: int = 3600):
        """The whole process for a SMAC based on ps-lite.
        港真没必要用ps-lite，但既然要求用，就试试看。

        Parameters
        ----------
        X_train : np.ndarray
        y_train : np.ndarray
        X_valid : np.ndarray
        y_valid : np.ndarray
        cs : ConfigurationSpace
            The configuration space for the hyper-parameters.
        temp_folder : str
            Folder that save the histories.
        scheduler_args L typing.List[str]
            Arguments of scheduler for ps-lite.
        server_args : typing.List[str]
            Arguments of server for ps-lite.
        worker_args : typing.List[str]
            Arguments of workers for ps-lite. It should be a list of args.
        our_work : str, default_value = None
            The path to the loss file in order to calculate the gradient for the
            DummyHOAG.
        cutoff : int, default_value = 3600
            Maximum runtime, after which the target algorithm is cancelled.
        """
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.cs = cs
        self.temp_folder = temp_folder
        self.scheduler_args = scheduler_args
        self.server_args = server_args
        self.worker_args = worker_args
        self.our_work = our_work
        self.cutoff = cutoff

    # 进程类固定名字run
    def run(self):
        """Run the whole PS_SMAC process.

        Returns
        -------
        """
        # 首先创建一堆worker，存放起来
        workers = []
        for worker_id in range(len(self.worker_args)):
            workers.append(self.create_worker(worker_id))
        smac = self.create_server()
        scheduler = self.create_scheduler()
        # 首先将每个worker全部打开，然后运行smac
        for worker in workers:
            worker.start()
        smac.optimize()
        # 最后等待worker结束(实际上并不会结束)
        for worker in workers:
            worker.join()

    def create_server(self) -> SMAC:
        """Function to create the server.

        Returns
        -------
        smac : SMAC
            Return the fully configured SMAC object.
        """
        # 首先创建输出文件夹
        output_dir = self.temp_folder + "server-output_%s" % (
            datetime.datetime.fromtimestamp(time.time()).strftime(
                '%Y-%m-%d_%H:%M:%S_%f'))

        # 创建scenario
        scenario_dict = {
            "cs": self.cs,
            "run_obj": "quality",
            "cutoff_time": self.cutoff,
            "initial_incumbent": "RANDOM",
            "output_dir": output_dir
        }
        scenario = Scenario(scenario_dict)

        # Runhistory实例
        runhistory = RunHistory(aggregate_func=average_cost)
        runhistory2epm = RunHistory2EPM4Cost(scenario=scenario,
                                             num_params=1,
                                             success_states=[
                                                 StatusType.SUCCESS,
                                                 StatusType.CRASHED],
                                             impute_censored_data=False,
                                             impute_state=None)

        # 创建server对象
        server = Server(self.server_args, self.cs)
        # 创建ta函数，不存放数据，因为server不真正运行ta
        ta = CustomizedTA(np.array(0), np.array(0), np.array(0), np.array(0))

        # 创建smac
        if self.our_work is not None:
            # our_work原则上是打开gradient文件路径的str
            hoag = self._cal_hoag()
            # 创建epm
            gpr = GaussianGradientEPM
            smac = SMAC(
                scenario=scenario,
                tae_runner=ta,
                model=gpr(),
                hoag=hoag,
                runhistory2epm=runhistory2epm,
                runhistory=runhistory,
                server=server
            )
        else:
            smac = SMAC(scenario=scenario, tae_runner=ta,
                        runhistory2epm=runhistory2epm,
                        runhistory=runhistory,
                        server=server)

        # 返回smac对象
        return smac

    def create_worker(self, worker_id) -> Worker:
        """Function to create a worker.

        Parameters
        ----------
        worker_id : int
            A seed for the random generator. It should be the id of the worker.

        Returns
        -------
        worker : Worker
            Return a ps-lite worker.
        """
        # 首先指定输出目录
        output_dir = self.temp_folder + "worker-output_%s" % (
            datetime.datetime.fromtimestamp(time.time()).strftime(
                '%Y-%m-%d_%H:%M:%S_%f'))
        # 然后创建scenario
        scenario_dict = {
            "cs": self.cs,
            "run_obj": "quality",
            "cutoff_time": self.cutoff,
            "initial_incumbent": "RANDOM",
            "output_dir": output_dir
        }
        scenario = Scenario(scenario_dict)

        # 统计类
        stats = Stats(scenario)
        # 创建ta函数
        ta = CustomizedTA(self.X_train, self.y_train, self.X_valid,
                          self.y_valid)
        tae_runner = ExecuteTAFuncDict(ta=ta,
                                       stats=stats,
                                       run_obj=scenario.run_obj,
                                       memory_limit=scenario.memory_limit,
                                       runhistory=RunHistory(
                                           aggregate_func=average_cost),
                                       par_factor=scenario.par_factor,
                                       cost_for_crash=scenario.cost_for_crash)
        # logger和rng
        traj_logger = TrajLogger(output_dir=output_dir, stats=stats)
        rng = np.random.RandomState(seed=worker_id)
        # 创建intensifier
        intensifier = Intensifier(tae_runner=tae_runner,
                                  stats=stats,
                                  traj_logger=traj_logger,
                                  rng=rng,
                                  instances=scenario.train_insts)

        # 最终目的，创建worker并返回
        worker = Worker(self.worker_args[worker_id], self.cs, intensifier,
                        worker_id=worker_id)
        return worker

    def create_scheduler(self) -> subprocess.Popen:
        """Function to create a scheduler.

        Returns
        -------
        sheduler : subprocess.Popen
            Return the scheduler.
        """
        # scheduler和server是主进程的子进程，worker是子进程的子进程
        scheduler = subprocess.Popen(self.scheduler_args, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        return scheduler

    def _cal_hoag(self) -> AbstractHOAG:
        """Calculate the dummy hoag.

        Returns
        -------
        hoag : AbstractHOAG
            Return the AbstractHOAG object.
        """
        # 读入梯度信息
        with open(self.our_work, "r") as fp:
            lines = fp.readlines()
        # 计算loss的值
        loss = [float(line.strip().split()[0]) for line in lines]
        gradient = []
        for i in range(1, len(loss)):
            gradient.append((loss[i] - loss[i - 1]) * len(loss))
        hoag = DummyHOAG(0.00095, 1, np.array(gradient))

        return hoag
