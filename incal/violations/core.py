import random

import numpy as np

from pywmi.smt_check import evaluate
from pywmi.smt_print import pretty_print, pretty_print_instance


class SelectionStrategy(object):
    def select_active(self, domain, data, labels, formula, active_indices):
        raise NotImplementedError()


class AllViolationsStrategy(SelectionStrategy):
    def select_active(self, domain, data, labels, formula, active_indices):
        active_set = set(active_indices)
        learned_labels = evaluate(domain, formula, data)
        differences = np.logical_xor(labels, learned_labels)
        difference_set = set(np.where(differences)[0])
        # print(active_set)
        # print(difference_set)
        # print(pretty_print(formula))
        # for i in active_set & difference_set:
        #     print(i)
        #     print(pretty_print_instance(domain, data[i]))
        #     print(labels[i], learned_labels[i])
        # print()
        # assert len(active_set & difference_set) == 0
        return sorted(difference_set - active_set)


class RandomViolationsStrategy(AllViolationsStrategy):
    def __init__(self, sample_size):
        self.sample_size = sample_size
        self.last_violations = None

    def select_active(self, domain, data, labels, formula, active_indices):
        all_violations = list(AllViolationsStrategy.select_active(self, domain, data, labels, formula, active_indices))
        self.last_violations = all_violations
        sample_size = min(self.sample_size, len(all_violations))
        return random.sample(all_violations, sample_size)


class WeightedRandomViolationsStrategy(AllViolationsStrategy):
    def __init__(self, sample_size, weights):
        self.sample_size = sample_size
        self.last_violations = None
        self.weights = weights

    def select_active(self, domain, data, labels, formula, active_indices):
        all_violations = list(AllViolationsStrategy.select_active(self, domain, data, labels, formula, active_indices))
        self.last_violations = all_violations
        sample_size = min(self.sample_size, len(all_violations))
        import sampling
        return sampling.sample_weighted(zip(all_violations, [self.weights[i] for i in all_violations]), sample_size)


class MaxViolationsStrategy(AllViolationsStrategy):
    def __init__(self, sample_size, weights):
        self.sample_size = sample_size
        self.last_violations = None
        self.weights = weights

    def select_active(self, domain, data, labels, formula, active_indices):
        all_violations = list(AllViolationsStrategy.select_active(self, domain, data, labels, formula, active_indices))
        self.last_violations = all_violations
        sample_size = min(self.sample_size, len(all_violations))
        weighted_violations = zip(all_violations, [self.weights[i] for i in all_violations])
        weighted_violations = sorted(weighted_violations, key=lambda t: t[1])
        return [t[0] for t in weighted_violations[0:sample_size]]
