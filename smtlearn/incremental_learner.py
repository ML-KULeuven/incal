import random
import time

from learner import Learner

import pysmt.shortcuts as smt

import observe
from smt_check import SmtChecker


class SelectionStrategy(object):
    def select_active(self, domain, data, formula, active_indices):
        raise NotImplementedError()

    @staticmethod
    def check_example(domain, formula, assignment):
        return SmtChecker(assignment).check(formula)


class AllViolationsStrategy(SelectionStrategy):
    def select_active(self, domain, data, formula, active_indices):
        active_set = set(active_indices)
        for i in range(len(data)):
            if i not in active_set:
                learned_label = SelectionStrategy.check_example(domain, formula, data[i][0])
                if learned_label != data[i][1]:
                    yield i


class RandomViolationsStrategy(AllViolationsStrategy):
    def __init__(self, sample_size):
        self.sample_size = sample_size
        self.last_violations = None

    def select_active(self, domain, data, formula, active_indices):
        all_violations = list(AllViolationsStrategy.select_active(self, domain, data, formula, active_indices))
        self.last_violations = all_violations
        sample_size = min(self.sample_size, len(all_violations))
        return random.sample(all_violations, sample_size)


class IncrementalObserver(observe.SpecializedObserver):
    def observe_initial(self, initial_indices):
        raise NotImplementedError()

    def observe_iteration(self, theory, new_active_indices, solving_time, selection_time):
        raise NotImplementedError()


class IncrementalLearner(Learner):
    def __init__(self, name, selection_strategy):
        """
        Initializes a new incremental learner
        :param str name: The learner name
        :param SelectionStrategy selection_strategy: The selection strategy
        """
        Learner.__init__(self, "incremental_{}".format(name))
        self.selection_strategy = selection_strategy
        self.observer = observe.DispatchObserver()

    def add_observer(self, observer):
        self.observer.add_observer(observer)

    def learn(self, domain, data, initial_indices=None):

        active_indices = list(range(len(data))) if initial_indices is None else initial_indices
        all_active_indices = active_indices

        self.observer.observe("initial", active_indices)

        formula = None

        with smt.Solver() as solver:
            while len(active_indices) > 0:
                solving_start = time.time()
                formula = self.learn_partial(solver, domain, data, active_indices)
                solving_time = time.time() - solving_start

                selection_start = time.time()
                new_active_indices = list(self.selection_strategy.select_active(domain, data, formula, all_active_indices))
                active_indices = new_active_indices
                all_active_indices += active_indices
                selection_time = time.time() - selection_start
                self.observer.observe("iteration", formula, active_indices, solving_time, selection_time)

        return formula

    def learn_partial(self, solver, domain, data, new_active_indices):
        raise NotImplementedError()
