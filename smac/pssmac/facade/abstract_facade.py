from multiprocessing import Process
import typing


class AbstractFacade(Process):
    def __init__(self,
                 ps_args: typing.List[str]):
        """Abstract facade class for PS-SMAC.
        其实没必要用PS-Lite，可以考虑py的轻量级ps，但用了也要弄好。

        Parameters
        ----------
        ps_args : typing.List[str]
            Arguments for PS-Lite (Server/Worker/Scheduler)
        """
        Process.__init__(self)
        self.ps_args = ps_args
        # 保存Node对应的内容，初始化为None，在init方法中进行初始化
        self.facade = None

    def init(self):
        """Virtual method for initialization.

        Returns
        -------
        """
        raise NotImplementedError

    def run(self):
        """start method for Process class.

        Returns
        -------
        """
        if self.facade is None:
            raise RuntimeError("Call init to initialize the facade first.")
