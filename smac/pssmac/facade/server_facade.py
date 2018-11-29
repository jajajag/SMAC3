from smac.epm.gaussian_gradient_epm import GaussianGradientEPM
from smac.epm.hoag import AbstractHOAG, DummyHOAG
from smac.facade.smac_facade import SMAC
from smac.optimizer.objective import average_cost
from smac.pssmac.facade.abstract_facade import AbstractFacade
from smac.pssmac.ps.server_ps import Server
from smac.pssmac.tae.abstract_tae import AbstractTAE
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
import datetime
import numpy as np
import time
import typing


class ServerFacade(AbstractFacade):
    def __init__(self,
                 ps_args: typing.List[str],
                 tae_runner: AbstractTAE,
                 temp_folder: str = "./tmp/",
                 our_work: str = None,
                 cutoff: int = 3600):
        """The whole process for a SMAC server based on ps-lite.

        Parameters
        ----------
        worker_args : typing.List[str]
            Arguments of workers for ps-lite. It should be a list of args.
        cs : ConfigurationSpace
            The configuration space for the hyper-parameters.
        temp_folder : str, default_value = "./tmp/"
            Folder that save the histories.
        our_work : str, default_value = None
            The path to the loss file in order to calculate the gradient for the
            DummyHOAG.
        cutoff : int, default_value = 3600
            Maximum runtime, after which the target algorithm is cancelled.
        """
        AbstractFacade.__init__(self, ps_args)
        self.tae_runner = tae_runner
        self.temp_folder = temp_folder
        self.our_work = our_work
        self.cutoff = cutoff

    def init(self) -> AbstractFacade:
        """Function to create the server.

        Returns
        -------
        self : AbstractFacade
            Return the class itself.
        """
        # 首先创建输出文件夹
        output_dir = self.temp_folder + "server-output_%s" % (
            datetime.datetime.fromtimestamp(time.time()).strftime(
                '%Y-%m-%d_%H:%M:%S_%f'))

        # 创建scenario
        scenario_dict = {
            "cs": self.tae_runner.get_config_space(),
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
        server = Server(self.ps_args, self.tae_runner.get_config_space())

        # 创建smac
        if self.our_work is not None:
            # our_work原则上是打开gradient文件路径的str
            hoag = self._cal_hoag()
            # 创建epm
            gpr = GaussianGradientEPM
            self.facade = SMAC(
                scenario=scenario,
                tae_runner=self.tae_runner,
                model=gpr(),
                hoag=hoag,
                runhistory2epm=runhistory2epm,
                runhistory=runhistory,
                server=server
            )
        else:
            # 创建smac对象
            self.facade = SMAC(scenario=scenario, tae_runner=self.tae_runner,
                               runhistory2epm=runhistory2epm,
                               runhistory=runhistory,
                               server=server)

        # 返回smac对象
        return self

    def run(self):
        """Run the whole PS_SMAC process.

        Returns
        -------
        """
        # 调用父类的run来判断是否初始化
        AbstractFacade.run(self)
        # 调用smac的optimize函数
        self.facade.optimize()

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
