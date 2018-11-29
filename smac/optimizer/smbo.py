import os
import logging
import numpy as np
import time
import typing

from smac.configspace import Configuration
from smac.epm.base_epm import AbstractEPM
from smac.epm.bayes_opt.bayesian_optimization import BayesianOptimization
from smac.epm.hoag.lr_hoag import AbstractHOAG
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.initial_design.initial_design import InitialDesign
from smac.intensification.intensification import Intensifier
from smac.optimizer import pSMAC
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown, \
    ChooserLinearCoolDown
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer, \
    RandomSearch
from smac.pssmac.ps.server_ps import Server
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run import FirstRunCrashedException
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.util_funcs import remove_same_values
from smac.utils.validate import Validator



__author__ = "Aaron Klein, Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"


class SMBO(object):

    """Interface that contains the main Bayesian optimization loop

    Attributes
    ----------
    logger
    incumbent
    scenario
    config_space
    stats
    initial_design
    runhistory
    rh2EPM
    intensifier
    aggregate_func
    num_run
    model
    acq_optimizer
    acquisition_func
    rng
    random_configuration_chooser
    """

    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 initial_design: InitialDesign,
                 runhistory: RunHistory,
                 runhistory2epm: AbstractRunHistory2EPM,
                 intensifier: Intensifier,
                 aggregate_func: callable,
                 num_run: int,
                 model: AbstractEPM,
                 acq_optimizer: AcquisitionFunctionMaximizer,
                 acquisition_func: AbstractAcquisitionFunction,
                 rng: np.random.RandomState,
                 restore_incumbent: Configuration=None,
                 random_configuration_chooser: typing.Union[
                     ChooserNoCoolDown,
                     ChooserLinearCoolDown]=ChooserNoCoolDown(2.0),
                 # 强行在smbo中加入训练集和验证集
                 hoag: AbstractHOAG = None,
                 # 参数服务器Server的class
                 server: Server = None,
                 bayesian_optimization: bool = False):
        """
        Interface that contains the main Bayesian optimization loop

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        stats: Stats
            statistics object with configuration budgets
        initial_design: InitialDesign
            initial sampling design
        runhistory: RunHistory
            runhistory with all runs so far
        runhistory2epm : AbstractRunHistory2EPM
            Object that implements the AbstractRunHistory2EPM to convert runhistory
            data into EPM data
        intensifier: Intensifier
            intensification of new challengers against incumbent configuration
            (probably with some kind of racing on the instances)
        aggregate_func: callable
            how to aggregate the runs in the runhistory to get the performance of a
             configuration
        num_run: int
            id of this run (used for pSMAC)
        model: AbstractEPM
            empirical performance model (right now, we support only AbstractEPM)
        acq_optimizer: AcquisitionFunctionMaximizer
            Optimizer of acquisition function.
        acquisition_function : AcquisitionFunction
            Object that implements the AbstractAcquisitionFunction (i.e., infill
            criterion for acq_optimizer)
        restore_incumbent: Configuration
            incumbent to be used from the start. ONLY used to restore states.
        rng: np.random.RandomState
            Random number generator
        random_configuration_chooser
            Chooser for random configuration -- one of
            * ChooserNoCoolDown(modulus)
            * ChooserLinearCoolDown(start_modulus, modulus_increment, end_modulus)
        hoag : AbstractHOAG, defualt_value = None,
            An AbstractHOAG class. Use this value to calculate gradients of
            the loss function.
        server: Server, default_value = None
            Server node for PS-Lite. Set this value to None if you do not
            want to use PS_SMAC. Otherwise, it should be set to a Server object.
        bayesian_optimization : bool, default_value = False
            Flag to determine whether to use bayesian optimization (GPR).
        """

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.incumbent = restore_incumbent

        self.scenario = scenario
        self.config_space = scenario.cs
        self.stats = stats
        self.initial_design = initial_design
        self.runhistory = runhistory
        self.rh2EPM = runhistory2epm
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func
        self.num_run = num_run
        self.model = model
        self.acq_optimizer = acq_optimizer
        self.acquisition_func = acquisition_func
        self.rng = rng
        self.random_configuration_chooser = random_configuration_chooser
        # hoag的类，直接使用hoag的fit，predict等
        self.hoag = hoag
        # 保存server端进程
        self.server = server
        self.bayesian_optimization = bayesian_optimization

        self._random_search = RandomSearch(
            acquisition_func, self.config_space, rng
        )

    def start(self):
        """Starts the Bayesian Optimization loop.
        Detects whether we the optimization is restored from previous state.
        """
        self.stats.start_timing()
        # Initialization, depends on input
        if self.stats.ta_runs == 0 and self.incumbent is None:
            if self.server is None:
                self.incumbent = self.initial_design.run()
            else:
                # 由worker自己产生第一个incumbent，然后由server接收其中的一个
                self.incumbent, new_runhistory = self.server.pull()
                self.runhistory.update(new_runhistory)

        elif self.stats.ta_runs > 0 and self.incumbent is None:
            raise ValueError("According to stats there have been runs performed, "
                             "but the optimizer cannot detect an incumbent. Did "
                             "you set the incumbent (e.g. after restoring state)?")
        elif self.stats.ta_runs == 0 and self.incumbent is not None:
            raise ValueError("An incumbent is specified, but there are no runs "
                             "recorded in the Stats-object. If you're restoring "
                             "a state, please provide the Stats-object.")
        else:
            # Restoring state!
            self.logger.info("State Restored! Starting optimization with "
                             "incumbent %s", self.incumbent)
            self.logger.info("State restored with following budget:")
            self.stats.print_stats()

        # To be on the safe side -> never return "None" as incumbent
        if not self.incumbent:
            self.incumbent = self.scenario.cs.get_default_configuration()

    def run(self):
        """Runs the Bayesian optimization loop

        Returns
        ----------
        incumbent: np.array(1, H)
            The best found configuration
        """
        self.start()
        # 设置一个counter
        counter = 0
        # Main BO loop
        while True:
            # 打印每轮SMBO的最优结果(包括首轮SMBO 0)
            print('SMBO ' + str(counter) + ': ' + str(
                self.runhistory.get_cost(self.incumbent)))
            counter += 1

            if self.scenario.shared_model:
                pSMAC.read(run_history=self.runhistory,
                           output_dirs=self.scenario.input_psmac_dirs,
                           configuration_space=self.config_space,
                           logger=self.logger)

            start_time = time.time()
            X, Y = self.rh2EPM.transform(self.runhistory)

            self.logger.debug("Search for next configuration")
            # get all found configurations sorted according to acq
            challengers = self.choose_next(X, Y)

            time_spent = time.time() - start_time
            time_left = self._get_timebound_for_intensification(time_spent)

            self.logger.debug("Intensify")

            if self.server is None:
                self.incumbent, inc_perf = self.intensifier.intensify(
                    challengers=challengers,
                    incumbent=self.incumbent,
                    run_history=self.runhistory,
                    aggregate_func=self.aggregate_func,
                    time_bound=max(self.intensifier._min_time, time_left))
            else:
                # 从worker读取loss，加入history再运行新的challengers
                self.server.push(incumbent=self.incumbent,
                                 runhistory=self.runhistory,
                                 challengers=challengers.challengers,
                                 time_left=time_left)
                # 从worker读取runhistory，并merge到self.runhistory
                incumbent, new_runhistory = self.server.pull()
                self.runhistory.update(new_runhistory)
                # 更新了runhistory之后，应该找寻是否存在新的incumbent
                # 因为worker没有完整的
                runhistory_old = self.runhistory.get_history_for_config(
                    self.incumbent)
                runhistory_new = self.runhistory.get_history_for_config(
                    incumbent)
                # 找寻cost最小值
                lowest_cost_old = min([cost[0] for cost in runhistory_old])
                lowest_cost_new = min([cost[0] for cost in runhistory_new])
                if lowest_cost_new < lowest_cost_old:
                    # 替换为新的incumbent
                    self.incumbent = incumbent
                """可以考虑用这个函数
                new_incumbent = self._compare_configs(
                    incumbent=incumbent, challenger=challenger,
                    run_history=run_history,
                    aggregate_func=aggregate_func,
                    log_traj=log_traj)
                """

            if self.scenario.shared_model:
                pSMAC.write(run_history=self.runhistory,
                            output_directory=self.scenario.output_dir_for_this_run,
                            logger=self.logger)

            logging.debug("Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                self.stats.get_remaing_time_budget(),
                self.stats.get_remaining_ta_budget(),
                self.stats.get_remaining_ta_runs()))

            if self.stats.is_budget_exhausted():
                break

            self.stats.print_stats(debug_out=True)

        return self.incumbent

    def choose_next(self, X: np.ndarray, Y: np.ndarray,
                    incumbent_value: float = None):
        """Choose next candidate solution with Bayesian optimization. The 
        suggested configurations depend on the argument ``acq_optimizer`` to
        the ``SMBO`` class.

        Parameters
        ----------
        X : (N, D) numpy array
            Each row contains a configuration and one set of
            instance features.
        Y : (N, O) numpy array
            The function values for each configuration instance pair.
        incumbent_value: float
            Cost value of incumbent configuration
            (required for acquisition function);
            if not given, it will be inferred from runhistory;
            if not given and runhistory is empty,
            it will raise a ValueError

        Returns
        -------
        Iterable
        """
        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations
            return self._random_search.maximize(
                runhistory=self.runhistory, stats=self.stats, num_points=1
            )

        # 消去完全相同的行
        X, Y = remove_same_values(X, Y)
        print(X.shape)

        # 如果指定了hoag函数，则进行调用
        if self.hoag is not None:

            # 初始化梯度数组
            gradient = np.zeros(X.shape)
            # 对每组X，计算对应的梯度(此处有大量重复计算)
            for i in range(X.shape[0]):
                self.hoag.fit(X[i, :])
                gradient[i, :] = self.hoag.predict_gradient()

            X = X.flatten()
            ind = np.argsort(X)
            gradient = gradient.flatten()[ind].reshape(-1, 1)
            X = X[ind].reshape(-1, 1)
            Y = Y.flatten()[ind].reshape(-1, 1)
            self.model.train(X, Y, gradient=gradient)

        elif self.bayesian_optimization:
            # gpr使用的参数
            gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
            # 从configspace读取超参的范围
            pbounds = {}
            for key in self.scenario.cs._hyperparameters.keys():
                # 只处理float类型的超参
                hyperparamter = self.scenario.cs._hyperparameters[key],
                if isinstance(hyperparamter.default_value, float):
                    pbounds[key] = (hyperparamter.lower, hyperparamter.upper)
            # 初始化bayesian_optimization
            bo = BayesianOptimization(X, Y, pbounds=pbounds, verbose=False)
            # 预测下一个ei取得点
            newX = bo.maximize(acq="ei", **gp_params)
            # 将超参数组再转化为Configuration
            challengers = [Configuration(self.scenario.cs, x) for x in newX]
            return challengers

        else:
            self.model.train(X, Y)
        # 打印X和Y的值
        # print("X: ", X.flatten())
        # print("Y: ", Y.flatten())
        # print("Y_pred: ", self.model.predict(X))
        # if self.hoag is not None:
        #    print("G: ", gradient)

        if incumbent_value is None:
            if self.runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of "
                                 "the incumbent is unknown.")
            incumbent_value = self.runhistory.get_cost(self.incumbent)

        self.acquisition_func.update(model=self.model, eta=incumbent_value)

        challengers = self.acq_optimizer.maximize(
            runhistory=self.runhistory, 
            stats=self.stats,
            # 初始为5000，提升速度调成500
            num_points=500,
            random_configuration_chooser=self.random_configuration_chooser
        )
        return challengers

    def validate(self, config_mode='inc', instance_mode='train+test',
                 repetitions=1, use_epm=False, n_jobs=-1, backend='threading'):
        """Create validator-object and run validation, using
        scenario-information, runhistory from smbo and tae_runner from intensify

        Parameters
        ----------
        config_mode: str or list<Configuration>
            string or directly a list of Configuration
            str from [def, inc, def+inc, wallclock_time, cpu_time, all]
            time evaluates at cpu- or wallclock-timesteps of:
            [max_time/2^0, max_time/2^1, max_time/2^3, ..., default]
            with max_time being the highest recorded time
        instance_mode: string
            what instances to use for validation, from [train, test, train+test]
        repetitions: int
            number of repetitions in nondeterministic algorithms (in
            deterministic will be fixed to 1)
        use_epm: bool
            whether to use an EPM instead of evaluating all runs with the TAE
        n_jobs: int
            number of parallel processes used by joblib

        Returns
        -------
        runhistory: RunHistory
            runhistory containing all specified runs
        """
        if isinstance(config_mode, str):
            traj_fn = os.path.join(self.scenario.output_dir_for_this_run, "traj_aclib2.json")
            trajectory = TrajLogger.read_traj_aclib_format(fn=traj_fn, cs=self.scenario.cs)
        else:
            trajectory = None
        if self.scenario.output_dir_for_this_run:
            new_rh_path = os.path.join(self.scenario.output_dir_for_this_run, "validated_runhistory.json")
        else:
            new_rh_path = None

        validator = Validator(self.scenario, trajectory, self.rng)
        if use_epm:
            new_rh = validator.validate_epm(config_mode=config_mode,
                                            instance_mode=instance_mode,
                                            repetitions=repetitions,
                                            runhistory=self.runhistory,
                                            output_fn=new_rh_path)
        else:
            new_rh = validator.validate(config_mode, instance_mode, repetitions,
                                        n_jobs, backend, self.runhistory,
                                        self.intensifier.tae_runner,
                                        output_fn=new_rh_path)
        return new_rh

    def _get_timebound_for_intensification(self, time_spent):
        """Calculate time left for intensify from the time spent on
        choosing challengers using the fraction of time intended for
        intensification (which is specified in
        scenario.intensification_percentage).

        Parameters
        ----------
        time_spent : float

        Returns
        -------
        time_left : float
        """
        frac_intensify = self.scenario.intensification_percentage
        if frac_intensify <= 0 or frac_intensify >= 1:
            raise ValueError("The value for intensification_percentage-"
                             "option must lie in (0,1), instead: %.2f" %
                             (frac_intensify))
        total_time = time_spent / (1 - frac_intensify)
        time_left = frac_intensify * total_time
        self.logger.debug("Total time: %.4f, time spent on choosing next "
                          "configurations: %.4f (%.2f), time left for "
                          "intensification: %.4f (%.2f)" %
                          (total_time, time_spent, (1 - frac_intensify), time_left, frac_intensify))
        return time_left
