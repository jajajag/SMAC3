from smac.pssmac.facade.abstract_facade import AbstractFacade
import subprocess
import typing


class SchedulerFacade(AbstractFacade):
    def __init__(self,
                 ps_args: typing.List[str]):
        """Class of scheduler for PS-Lite.

        Parameters
        ----------
        ps_args : typing.List[str]
            Arguments for PS-Lite (Server/Worker/Scheduler)
        """
        AbstractFacade.__init__(self, ps_args)

    def init(self) -> AbstractFacade:
        """Function to create a scheduler.

        Returns
        -------
        sheduler : subprocess.Popen
            Return the scheduler.
        """
        # 创建一个scheduler进程用于协调server和worker
        self.facade = subprocess.Popen(self.ps_args, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE)
        return self

    def run(self):
        """Run the scheduler.

        Returns
        -------
        """
        AbstractFacade.run(self)
        # 等待直至scheduler线程执行完毕
        self.facade.wait()
