import random

import numpy as np
from incal.observe.inc_logging import LoggingObserver
from pysmt.fnode import FNode
from pywmi import smt_to_nested
from pywmi.domain import Density, Domain
from typing import Tuple, Optional

from .parameter_free_learner import learn_bottom_up
from .violations.core import RandomViolationsStrategy
from .violations.dt_selection import DecisionTreeSelection
from .k_cnf_smt_learner import KCnfSmtLearner
from .util.options import Options, Results


class LearnOptions(Options):
    def __init__(self):
        super().__init__(learn)
        self.add_option("domain", str, None, LearnOptions.domain_extraction)
        self.add_option("data", str, None, LearnOptions.np_extraction)
        self.add_option("labels", str, None, LearnOptions.np_extraction)

        self.add_option("learner", (str, str), ("cnf", "-"), Options.convert_dict(
            cnf=LearnOptions.cnf_factory_wrap
        ), arg_name="learner_factory")
        self.add_option("initial_strategy", (str, int), ("random", 20), Options.convert_dict(
            random=LearnOptions.initial_random
        ))
        self.add_option("selection_strategy", (str, int), ("random", 10), Options.convert_dict(
            random=LearnOptions.select_random,
            dt=LearnOptions.select_dt
        ))
        self.add_option("initial_k", int, 1)
        self.add_option("initial_h", int, 0)
        self.add_option("weight_k", float, 1)
        self.add_option("weight_h", float, 1)
        self.add_option("log", str)
        # self.add_option("max_k", int, None)
        # self.add_option("max_h", int, None)

    @staticmethod
    def domain_extraction(filename):
        return Density.import_from(filename).domain

    @staticmethod
    def np_extraction(filename):
        return np.load(filename)

    @staticmethod
    def cnf_factory_wrap(symmetries):
        def cnf_factory(k, h, selection_strategy):
            return KCnfSmtLearner(k, h, selection_strategy, symmetries=symmetries)
        return cnf_factory

    @staticmethod
    def initial_random(count):
        def random_selection(indices):
            return random.sample(indices, count)
        return random_selection

    @staticmethod
    def select_random(count):
        return RandomViolationsStrategy(count)

    @staticmethod
    def select_dt(count):
        return DecisionTreeSelection()

    def make_copy(self):
        return LearnOptions()


class LearnResults(Results):
    def __init__(self):
        super().__init__()
        self.add_duration()
        self.add_result("formula", LearnResults.extract_formula)
        self.add_result("k", LearnResults.extract_k)
        self.add_result("h", LearnResults.extract_h)

    @staticmethod
    def extract_formula(result):
        return smt_to_nested(result[0])

    @staticmethod
    def extract_k(result):
        return result[1]

    @staticmethod
    def extract_h(result):
        return result[2]


def learn(
        domain: Domain,
        data: np.ndarray,
        labels: np.ndarray,
        learner_factory: callable,
        initial_strategy: callable,
        selection_strategy: object,
        initial_k: int,
        initial_h: int,
        weight_k: float,
        weight_h: float,
        log: Optional[str]=None
) -> Tuple[FNode, int, int]:
    """
    Learn a formula that separates the positive and negative examples
    :return: A tuple containing 1. the learned formula, 2. the number of terms (or clauses) used,
    3. the number of hyperplanes used
    """

    # log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "results")
    # problem_name = hashlib.sha256(name).hexdigest()

    def learn_inc(_data, _labels, _i, _k, _h):
        learner = learner_factory(_k, _h, selection_strategy)
        initial_indices = initial_strategy(list(range(len(data))))
        # log_file = os.path.join(log_dir, "{}_{}_{}.txt".format(problem_name, _k, _h))
        if log is not None:
            learner.add_observer(LoggingObserver(log, _k, _h, None, False, selection_strategy))
        return learner.learn(domain, _data, _labels, initial_indices)

    return learn_bottom_up(data, labels, learn_inc, weight_k, weight_h, initial_k, initial_h, None, None)
