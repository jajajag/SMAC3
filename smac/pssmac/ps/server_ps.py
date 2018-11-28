from smac.configspace import Configuration, ConfigurationSpace
from smac.optimizer.objective import average_cost
from smac.pssmac.ps.abstract_ps import AbstractPS, ConfigHistory
from smac.runhistory.runhistory import RunHistory
import typing


class Server(AbstractPS):
    def __init__(self,
                 ps_args: typing.List[str],
                 cs: ConfigurationSpace,
                 aggregate_func: callable = average_cost) -> None:
        """Initialize Server.

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
        # 显式调用初始化self.ps
        AbstractPS.__init__(self, ps_args, cs, aggregate_func=aggregate_func)

    def push_parser(self, **kwargs) -> typing.Tuple[float, typing.List[str]]:
        """Parse incumbent, its runhistory and challengers to a string.

        Parameters
        ----------
        kwargs["incumbent"] : Configuration
            The current best Configuration.
        kwargs["runhistory"] : Runhistory
            The Runhistory contains only the incumbent value.
        kwargs["challengers"] : typing.List[Configuration]
            List of challengers Configuration.

        Returns
        -------
        ret_list : typing.List[str]
            Return the string to be passed to the PS-Lite server.
        """
        # 如果缺少输入参数，则报错
        if not ("incumbent" in kwargs
                and "runhistory" in kwargs
                and "challengers" in kwargs
                and "time_left" in kwargs):
            raise KeyError("At least one of the arguments: [incumbent, \
                    runhistory, challengers, time_left] are missing.")

        # 将incumbent和challengers分别赋值为对应的字符串。
        # 其中challengers不需要指定Runhistory
        incumbent = ConfigHistory(kwargs["incumbent"], self.cs,
                                  runhistory=kwargs["runhistory"]).to_str()
        challengers = [ConfigHistory(challenger, self.cs).to_str() for
                       challenger in
                       kwargs["challengers"]]
        time_left = kwargs["time_left"]

        # 返回由config/runhistory的str组成的list，已经time_left
        return time_left, [incumbent] + challengers

    def pull_parser(self, time_left: float, data: typing.List[str]) -> \
            typing.Tuple[Configuration, RunHistory]:
        """Parse the results from intensifier to incumbent and the runhistory.

        Parameters
        ----------
        time_left : float
            Time left for the smbo(useless here).
        data : typing.List[str]
            List of strings that contains configurations and runhistory info.

        Returns
        -------
        incumbent : Configuration
            Current best configuration.
        runhistory : RunHistory
            The new runhistory after judging some challengers
        """
        # 读取第一行，为incumbent
        incumbent = ConfigHistory.read_str(data[0], self.cs,
                                           aggregate_func=self.aggregate_func)
        runhistory = incumbent.get_runhistory()

        # 剩下行只需要更新runhistory就行了
        for line in data[1:]:
            config_history = ConfigHistory.read_str(
                line, self.cs, aggregate_func=self.aggregate_func)
            runhistory.update(config_history.get_runhistory())

        # 返回incumbent的Configuration以及全部的runhistory
        return incumbent.get_config(), runhistory
