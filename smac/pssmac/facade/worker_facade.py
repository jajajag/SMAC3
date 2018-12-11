from smac.intensification.intensification import Intensifier
from smac.optimizer.objective import average_cost
from smac.pssmac.facade.abstract_facade import AbstractFacade
from smac.pssmac.ps.worker_ps import Worker
from smac.pssmac.tae.abstract_tae import AbstractTAE
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.utils.io.traj_logging import TrajLogger
import datetime
import numpy as np
import time
import typing


class WorkerFacade(AbstractFacade):
    def __init__(self,
                 ps_args: typing.List[str]):
        """Create a PS-Lite worker node.

        Parameters
        ----------
        ps_args : typing.List[str]
            Arguments passed to the PS-Lite worker node.
        """
        AbstractFacade.__init__(self, ps_args)
        # 提前创建worker
        self.facade = Worker(self.ps_args)
        self.tae_runner = None
        self.temp_folder = "./tmp/"
        self.worker_id = 0
        self.per_run_time_limit = 3600
        self.total_time_limit = 24 * 3600

    def init(self, **kwargs) -> AbstractFacade:
        """Function to create a worker.

        Parameters
        ----------
        kwargs["tae_runner"]: AbstractTAE
            The ta function to train and fit the models.
        kwargs["temp_folder"] : str, default_value = "./tmp/"
            Folder that save the histories.
        kwargs["worker_id"] : int, default_value = 0
            Id of each worker. This value is used as seed of initial design,
            which should be unique.
        kwargs["per_run_time_limit"] : int, default_value = 3600
            Time limit of a single run of intensifier.
        kwargs["total_time_limit"] : int, default_value = 24 * 3600
            Total time limit for the SMBO process.

        Returns
        -------
        self : AbstractFacade
            Return the class itself.
        """
        # 初始化各个参数
        if "tae_runner" not in kwargs:
            raise AttributeError("Please specify a tae_runner for the facade.")
        self.tae_runner = kwargs["tae_runner"]
        if "temp_folder" in kwargs:
            self.temp_folder = kwargs["temp_folder"]
        if "worker_id" in kwargs:
            self.worker_id = kwargs["worker_id"]
        if "per_run_time_limit" in kwargs:
            self.per_run_time_limit = kwargs["per_run_time_limit"]
        if "total_time_limit" in kwargs:
            self.total_time_limit = kwargs["total_time_limit"]

        # 首先指定输出目录
        output_dir = self.temp_folder + "worker-output_%s" % (
            datetime.datetime.fromtimestamp(time.time()).strftime(
                '%Y-%m-%d_%H:%M:%S_%f'))

        # 然后创建scenario
        scenario_dict = {
            "cs": self.tae_runner.get_config_space(),
            "run_obj": "quality",
            # worker只保留单次运行的时间
            "cutoff_time": self.per_run_time_limit,
            "initial_incumbent": "RANDOM",
            "output_dir": output_dir
        }
        scenario = Scenario(scenario_dict)

        # 统计类
        stats = Stats(scenario)
        tae_runner = ExecuteTAFuncDict(ta=self.tae_runner,
                                       stats=stats,
                                       run_obj=scenario.run_obj,
                                       memory_limit=scenario.memory_limit,
                                       runhistory=RunHistory(
                                           aggregate_func=average_cost),
                                       par_factor=scenario.par_factor,
                                       cost_for_crash=scenario.cost_for_crash)
        # logger和rng
        traj_logger = TrajLogger(output_dir=output_dir, stats=stats)
        rng = np.random.RandomState(seed=self.worker_id)
        # 创建intensifier
        intensifier = Intensifier(tae_runner=tae_runner,
                                  stats=stats,
                                  traj_logger=traj_logger,
                                  rng=rng,
                                  instances=scenario.train_insts)

        # 对worker进行初始化
        self.facade.init(self.tae_runner.get_config_space(),
                         intensifier=intensifier,
                         worker_id=self.worker_id,
                         per_run_time_limit=self.per_run_time_limit,
                         total_time_limit=self.total_time_limit)

        return self

    def run(self):
        AbstractFacade.run(self)
        self.facade.start()
        start_time = time.time()
        while time.time() - start_time < 1.2 * self.total_time_limit:
            # 如果时间没到，但已经结束了，仅返回
            if not self.facade.is_alive():
                return
            # 等待一定时间
            time.sleep(int(self.total_time_limit / 100) + 1)
        # 时间到了，还未结束，则强制结束
        self.facade.end()
        self.facade.terminate()
        # self.facade.join()
