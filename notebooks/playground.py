from inspect import signature

import numpy as np

from smtlearn.examples import ice_cream_problem
from pywmi.plot import plot_data, plot_formula
from pywmi.sample import uniform
from pywmi.smt_check import evaluate
import random
from smtlearn.violations.core import RandomViolationsStrategy
from smtlearn.k_cnf_smt_learner import KCnfSmtLearner
from pywmi.smt_print import pretty_print

random.seed(666)
np.random.seed(666)

domain, formula, name = ice_cream_problem()
# plot_formula(None, domain, formula)

data = uniform(domain, 100)
labels = evaluate(domain, formula, data)

learner = KCnfSmtLearner(3, 3, RandomViolationsStrategy(10))
initial_indices = random.sample(range(data.shape[0]), 20)

learned_theory = learner.learn(domain, data, labels, initial_indices)
print(pretty_print(learned_theory))
