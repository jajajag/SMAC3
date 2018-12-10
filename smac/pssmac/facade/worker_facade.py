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
                 ps_args: typing.List[str],
                 tae_runner: AbstractTAE,
                 temp_folder: str = "./tmp/",
                 worker_id: int = 0,
                 per_run_time_limit: int = 3600
                 ):
        """Create a PS-Lite worker node.

        Parameters
        ----------
        ps_args : typing.List[str]
            Arguments passed to the PS-Lite worker node.
        tae_runner : AbstractTAE
            The runner of the model to be optimized.
        temp_folder : str, default_value = "./tmp/"
            Folder to save temporary files.
        worker_id : int, default_value = 0
            Id of each worker. This value is used as seed of initial design,
            which should be unique.
        cutoff : int, default_value = 3600
            Time limit of the whole smbo process.
        """
        AbstractFacade.__init__(self, ps_args)
        self.tae_runner = tae_runner
        self.temp_folder = temp_folder
        self.worker_id = worker_id
        self.per_run_time_limit = per_run_time_limit

    def init(self) -> AbstractFacade:
        """Function to create a worker.

        Returns
        -------
        self : AbstractFacade
            Return the class itself.
        """
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

        # 最终目的，创建worker并返回
        self.facade = Worker(self.ps_args, self.tae_runner.get_config_space(),
                             intensifier, worker_id=self.worker_id)
        return self

    def run(self):
        AbstractFacade.run(self)
        self.facade.start()
        self.facade.join()
