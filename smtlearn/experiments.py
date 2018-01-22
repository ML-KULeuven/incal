from __future__ import print_function

import random

import os

import plotting
from generator import import_synthetic_data_files
from inc_logging import LoggingObserver
from incremental_learner import AllViolationsStrategy, RandomViolationsStrategy
from k_cnf_smt_learner import KCnfSmtLearner

data_sets = list(import_synthetic_data_files("../data", "synthetic"))
selection_strategy = RandomViolationsStrategy(5)

seed = 666
random.seed(seed)

for data_set in data_sets:
    name = data_set.synthetic_problem.theory_problem.name
    if data_set.synthetic_problem.bool_count == 0 and data_set.synthetic_problem.real_count == 2\
            and data_set.synthetic_problem.terms_per_formula == 4:
        print(name)

        synthetic_problem = data_set.synthetic_problem
        initial_indices = random.sample(list(range(len(data_set.samples))), 20)
        learner = KCnfSmtLearner(synthetic_problem.formula_count, synthetic_problem.half_space_count, selection_strategy)

        dir_name = "../output/{}".format(name)
        exp_id = "{}_{}_{}".format(learner.name, len(data_set.samples), seed)
        learner.add_observer(plotting.PlottingObserver(data_set.samples, dir_name, exp_id, "r0", "r1"))
        learner.add_observer(LoggingObserver("{}{}{}.txt".format(dir_name, os.path.sep, exp_id), True, selection_strategy))
        learner.learn(synthetic_problem.theory_problem.domain, data_set.samples, initial_indices)
