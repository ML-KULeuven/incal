import time

import pysmt.shortcuts as smt

from .observe import observe
from .learner import Learner


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

    def learn(self, domain, data, labels, initial_indices=None):

        active_indices = list(range(len(data))) if initial_indices is None else initial_indices
        all_active_indices = active_indices

        self.observer.observe("initial", active_indices)

        formula = None

        with smt.Solver() as solver:
            while len(active_indices) > 0:
                solving_start = time.time()
                formula = self.learn_partial(solver, domain, data, labels, active_indices)
                solving_time = time.time() - solving_start

                selection_start = time.time()
                new_active_indices = list(self.selection_strategy.select_active(domain, data, formula, all_active_indices))
                active_indices = new_active_indices
                all_active_indices += active_indices
                selection_time = time.time() - selection_start
                self.observer.observe("iteration", formula, active_indices, solving_time, selection_time)

        return formula

    def learn_partial(self, solver, domain, data, labels, new_active_indices):
        raise NotImplementedError()
