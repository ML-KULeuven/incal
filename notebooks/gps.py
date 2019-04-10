import os
import random
import time

from pywmi import Domain, RejectionEngine, evaluate, plot

import numpy as np

from incal import Formula
from incal.violations.virtual_data import OneClassStrategy
from incal.k_cnf_smt_learner import KCnfSmtLearner
from incal.learn import LearnOptions
from incal.observe.plotting import PlottingObserver
from incal.parameter_free_learner import learn_bottom_up
from incal.violations.dt_selection import DecisionTreeSelection
from incal.violations.core import RandomViolationsStrategy


def experiment():
    random.seed(888)
    np.random.seed(888)

    start = time.time()

    domain = Domain.make([], ["x", "y"], [(0, 1), (0, 1)])
    x, y = domain.get_symbols(domain.variables)
    thresholds = {"x": 0.1, "y": 0.1}
    # data, _ = RejectionEngine(domain, formula, x * x, 100000).get_samples(50)
    filename = "/Users/samuelkolb/Downloads/gps-trajectory-simpler/cell-0-2.data"
    data = np.loadtxt(filename, delimiter=",", skiprows=0, usecols=[0, 1])
    minimum_0, minimum_1 = min(data[:, 0]), min(data[:, 1])
    maximum_0, maximum_1 = max(data[:, 0]), max(data[:, 1])
    data[:, 0] = (data[:, 0] - minimum_0) / (maximum_0 - minimum_0)
    data[:, 1] = (data[:, 1] - minimum_1) / (maximum_1 - minimum_1)
    k = 3
    nearest_neighbors = []

    for i in range(len(data)):
        nearest_neighbors.append([])
        for j in range(len(data)):
            if i != j:
                distance = 1 if any(data[i, b] != data[j, b] for b, v in enumerate(domain.variables)
                                    if domain.is_bool(v))\
                    else max(abs(data[i, r] - data[j, r]) / (domain.var_domains[v][1] - domain.var_domains[v][0]) for r, v in enumerate(domain.variables) if domain.is_real(v))
                if len(nearest_neighbors[i]) < k:
                    nearest_neighbors[i].append((j, distance))
                else:
                    index_of_furthest = None
                    for fi, f in enumerate(nearest_neighbors[i]):
                        if index_of_furthest is None or f[1] > nearest_neighbors[i][index_of_furthest][1]:
                            index_of_furthest = fi
                    if distance < nearest_neighbors[i][index_of_furthest][1]:
                        nearest_neighbors[i][index_of_furthest] = (j, distance)
    print(nearest_neighbors)
    t = [[sum(n[1] for n in nearest_neighbors[i]) / len(nearest_neighbors[i]) * (domain.var_domains[v][1] - domain.var_domains[v][0]) for v in domain.real_vars]
         for i in range(len(nearest_neighbors))]
    t = np.array(t) * 2
    print(t)
    # data = uniform(domain, 400)
    labels = np.ones(len(data))
    data, labels = OneClassStrategy.add_negatives(domain, data, labels, t, 1000)

    directory = "output{}gps{}{}".format(os.path.sep, os.path.sep, time.strftime("%Y-%m-%d %Hh%Mm%Ss"))
    os.makedirs(directory)

    name = os.path.join(directory, "combined.png")
    plot.plot_combined("x", "y", domain, None, (data, labels), None, name, set(), set())

    def learn_inc(_data, _labels, _i, _k, _h):
        # strategy = OneClassStrategy(RandomViolationsStrategy(10), thresholds)
        strategy = RandomViolationsStrategy(10)
        learner = KCnfSmtLearner(_k, _h, strategy, "mvn")
        initial_indices = LearnOptions.initial_random(20)(list(range(len(_data))))
        learner.add_observer(PlottingObserver(domain, directory, "run_{}_{}_{}".format(_i, _k, _h),
                                              domain.real_vars[0], domain.real_vars[1], None, False))
        return learner.learn(domain, _data, _labels, initial_indices)

    (new_data, new_labels, learned_formula), k, h = learn_bottom_up(data, labels, learn_inc, 1, 1, 4, 8, None, None)
    duration = time.time() - start
    Formula(domain, learned_formula).to_file(os.path.join(directory, "result_{}_{}_{}.json".format(k, h, int(duration))))


if __name__ == '__main__':
    experiment()