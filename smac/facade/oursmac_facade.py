from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from multiprocessing import Process
from smac.epm.gaussian_gradient_epm import GaussianGradientEPM
from smac.facade.smac_facade import SMAC
from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.tae.execute_ta_customized import CustomizedTA
from smac.epm.hoag.dummy_hoag import DummyHOAG
from smac.pssmac.ps.server_ps import Server
import numpy as np
import typing


class OurSMAC(Process):
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_valid: np.ndarray,
                 y_valid: np.ndarray,
                 dirs: typing.List[str],
                 smbo_id: int,
                 server: Server = None,
                 cs: ConfigurationSpace = None,
                 our_work: bool = False):
        if smbo_id >= len(dirs):
            raise ValueError("SMBO ID exceeds size of the pool.")
        super().__init__()
        # 创建一个ta函数
        ta = CustomizedTA(X_train, y_train, X_valid, y_valid)
        # 如果未指定configspace，则赋值默认
        if cs is None:
            cs = ConfigurationSpace()
            # 超参搜索空间，使用[1e-6, 1]
            alpha = UniformFloatHyperparameter(name="alpha", lower=1e-3,
                                               upper=1,
                                               default_value=1, log=False)
            cs.add_hyperparameters([alpha])

        # 创建scenario
        scenario_dict = {
            "cs": cs,
            "run_obj": "quality",
            "cutoff_time": 100,
            "shared_model": True if len(dirs) > 1 else False,
            #"initial_incumbent": 'RANDOM"
            "input_psmac_dirs": dirs,
            "output_dir": dirs[smbo_id]
        }

        scenario = Scenario(scenario_dict)

        runhistory = RunHistory(aggregate_func=average_cost)
        runhistory2epm = RunHistory2EPM4Cost(scenario=scenario,
                                             num_params=1,
                                             success_states=[
                                                 StatusType.SUCCESS,
                                                 StatusType.CRASHED],
                                             impute_censored_data=False,
                                             impute_state=None)

        # 创建smac
        if our_work is not None:
            # 读入梯度信息
            with open(our_work, "r") as fp:
                lines = fp.readlines()
            # 计算loss的值
            loss = [float(line.strip().split()[0]) for line in lines]
            gradient = []
            for i in range(1, len(loss)):
                gradient.append((loss[i] - loss[i - 1]) * len(loss))
            hoag = DummyHOAG(0.00095, 1, np.array(gradient))
            # 创建epm
            gpr = GaussianGradientEPM
            self.smac = SMAC(
                scenario=scenario,
                tae_runner=ta,
                model=gpr(),
                hoag=hoag,
                runhistory2epm=runhistory2epm,
                runhistory=runhistory,
                server=server
            )
        else:
            self.smac = SMAC(scenario=scenario, tae_runner=ta,
                             runhistory2epm=runhistory2epm,
                             runhistory=runhistory,
                             server=server)

    # 进程类固定名字run
    def run(self):
        self.smac.optimize()