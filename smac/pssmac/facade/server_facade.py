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
                 ps_args: typing.List[str]):
        """The whole process for a SMAC server based on ps-lite.

        Parameters
        ----------
        ps_args : typing.List[str]
            Arguments of workers for ps-lite. It should be a list of args.
        """
        AbstractFacade.__init__(self, ps_args)
        # 提前创建server对象
        self.server = Server(self.ps_args)
        self.tae_runner = None
        self.temp_folder = "./tmp/"
        self.our_work = None
        self.total_time_limit = 24 * 3600

    def init(self, **kwargs) -> AbstractFacade:
        """Function to create the server.

        Parameters
        ----------
        kwargs["tae_runner"]: AbstractTAE
            The ta function to train and fit the models.
        kwargs["temp_folder"] : str, default_value = "./tmp/"
            Folder that save the histories.
        kwargs["our_work"] : str, default_value = None
            The path to the loss file in order to calculate the gradient for the
            DummyHOAG.
        kwargs["total_time_limit"] : int, default_value = 24 * 3600
            Maximum runtime, after which the target algorithm is cancelled.

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
        if "our_work" in kwargs:
            self.our_work = kwargs["our_work"]
        if "total_time_limit" in kwargs:
            self.total_time_limit = kwargs["total_time_limit"]

        # 首先创建输出文件夹
        output_dir = self.temp_folder + "server-output_%s" % (
            datetime.datetime.fromtimestamp(time.time()).strftime(
                '%Y-%m-%d_%H:%M:%S_%f'))

        # 创建scenario
        scenario_dict = {
            "cs": self.tae_runner.get_config_space(),
            "run_obj": "quality",
            # smbo运行的总时间
            "wallclock_limit": self.total_time_limit,
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

        # 初始化server
        self.server.init(self.tae_runner.get_config_space())

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
                server=self.server
            )
        else:
            # 创建smac对象
            self.facade = SMAC(scenario=scenario, tae_runner=self.tae_runner,
                               runhistory2epm=runhistory2epm,
                               runhistory=runhistory,
                               server=self.server)

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

        Parameters
        ----------
        our_work : str, default_value = None
            The path to the loss file in order to calculate the gradient for the
            DummyHOAG.

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
