from __future__ import print_function

import argparse
import json
import random

import os

import time

import plotting
from generator import import_synthetic_data_files
from inc_logging import LoggingObserver
from incremental_learner import AllViolationsStrategy, RandomViolationsStrategy
from k_cnf_smt_learner import KCnfSmtLearner
from k_dnf_smt_learner import KDnfSmtLearner


def learn_synthetic(input_dir, prefix, results_dir, bias):
    input_dir = os.path.abspath(input_dir)
    data_sets = list(import_synthetic_data_files(input_dir, prefix))

    overview = os.path.join(results_dir, "problems.txt")

    if not os.path.isfile(overview):
        flat = {}
    else:
        with open(overview, "r") as f:
            flat = json.load(f)

    for data_set in data_sets:
        synthetic_problem = data_set.synthetic_problem
        name = synthetic_problem.theory_problem.name

        if name not in flat:
            flat[name] = {}

        print(name)

        sample_count = len(data_set.samples)

        seed = hash(time.time())
        random.seed(seed)

        initial_indices = random.sample(list(range(sample_count)), 20)
        h = synthetic_problem.half_space_count
        k = synthetic_problem.formula_count

        if bias == "cnf" or bias == "dnf":
            selection_strategy = RandomViolationsStrategy(10)
            if bias == "cnf":
                learner = KCnfSmtLearner(k, h, selection_strategy)
            elif bias == "dnf":
                learner = KDnfSmtLearner(k, h, selection_strategy)

            if synthetic_problem.bool_count == 0 and synthetic_problem.real_count == 2:
                feats = synthetic_problem.theory_problem.domain.real_vars
                plots_dir = os.path.join(results_dir, name)
                exp_id = "{}_{}_{}".format(learner.name, sample_count, seed)
                learner.add_observer(plotting.PlottingObserver(data_set.samples, plots_dir, exp_id, *feats))
            log_file = "{}_{}_{}_{}_{}.learning_log.txt".format(name, sample_count, seed, k, h)
            learner.add_observer(LoggingObserver(os.path.join(results_dir, log_file), seed, True, selection_strategy))
        else:
            raise RuntimeError("Unknown bias {}".format(bias))

        learner.learn(synthetic_problem.theory_problem.domain, data_set.samples, initial_indices)
        flat[name][sample_count] = {"k": k, "h": h, "seed": seed, "bias": bias}

    with open(overview, "w") as f:
        json.dump(flat, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("prefix")
    parser.add_argument("output_dir")
    parser.add_argument("bias")
    parsed = parser.parse_args()
    learn_synthetic(parsed.input_dir, parsed.prefix, parsed.output_dir, parsed.bias)
