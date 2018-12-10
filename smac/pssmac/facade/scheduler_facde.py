from smac.pssmac.facade.abstract_facade import AbstractFacade
import subprocess
import typing
import time


class SchedulerFacade(AbstractFacade):
    def __init__(self,
                 ps_args: typing.List[str],
                 total_time_limit: int = 24 * 3600):
        """Class of scheduler for PS-Lite.

        Parameters
        ----------
        ps_args : typing.List[str]
            Arguments for PS-Lite (Server/Worker/Scheduler)
        """
        AbstractFacade.__init__(self, ps_args)
        self.total_time_limit = total_time_limit

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
        # 运行1.2倍的总时间，如果不够的话，再加长之
        time.sleep(self.total_time_limit * 1.2)
        # 等待直至scheduler线程执行完毕
        self.facade.wait()
