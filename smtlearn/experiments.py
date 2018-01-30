from __future__ import print_function

import argparse
import json
import random

import os

import time

from generator import import_synthetic_data_files
from inc_logging import LoggingObserver
from incremental_learner import AllViolationsStrategy, RandomViolationsStrategy
from k_cnf_smt_learner import KCnfSmtLearner
from k_dnf_smt_learner import KDnfSmtLearner
from parameter_free_learner import learn_bottom_up
from timeout import timeout


def learn_synthetic(input_dir, prefix, results_dir, bias, plot=None, sample_count=None, time_out=None, non_inc=False,
                    parameter_free=False):
    input_dir = os.path.abspath(input_dir)
    data_sets = list(import_synthetic_data_files(input_dir, prefix))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    overview = os.path.join(results_dir, "problems.txt")

    if not os.path.isfile(overview):
        flat = {}
    else:
        with open(overview, "r") as f:
            flat = json.load(f)

    for data_set in data_sets:
        synthetic_problem = data_set.synthetic_problem
        data = data_set.samples
        name = synthetic_problem.theory_problem.name

        if name not in flat:
            flat[name] = {}

        print(name)

        seed = hash(time.time())
        random.seed(seed)

        if sample_count is not None and sample_count < len(data):
            data = random.sample(data, sample_count)
        else:
            sample_count = len(data)

        if not parameter_free:
            initial_indices = None if non_inc else random.sample(list(range(sample_count)), 20)
            h = synthetic_problem.half_space_count
            k = synthetic_problem.formula_count
            domain = synthetic_problem.theory_problem.domain

            if bias == "cnf" or bias == "dnf":
                selection_strategy = RandomViolationsStrategy(10)
                if bias == "cnf":
                    learner = KCnfSmtLearner(k, h, selection_strategy)
                elif bias == "dnf":
                    learner = KDnfSmtLearner(k, h, selection_strategy)

                if plot is not None and plot and synthetic_problem.bool_count == 0 and synthetic_problem.real_count == 2:
                    import plotting
                    feats = domain.real_vars
                    plots_dir = os.path.join(results_dir, name)
                    exp_id = "{}_{}_{}".format(learner.name, sample_count, seed)
                    learner.add_observer(plotting.PlottingObserver(data, plots_dir, exp_id, *feats))
                log_file = "{}_{}_{}_{}_{}.learning_log.txt".format(name, sample_count, seed, k, h)
                learner.add_observer(LoggingObserver(os.path.join(results_dir, log_file), seed, True, selection_strategy))
            else:
                raise RuntimeError("Unknown bias {}".format(bias))

            result = timeout(learner.learn, [domain, data, initial_indices], duration=time_out)
        else:
            def learn_f(_data, _k, _h):
                selection_strategy = RandomViolationsStrategy(10)
                if bias == "cnf":
                    learner = KCnfSmtLearner(_k, _h, selection_strategy)
                elif bias == "dnf":
                    learner = KDnfSmtLearner(_k, _h, selection_strategy)
                initial_indices = None if non_inc else random.sample(list(range(sample_count)), 20)
                log_file = "{}_{}_{}_{}_{}.learning_log.txt".format(name, sample_count, seed, _k, _h)
                learner.add_observer(LoggingObserver(os.path.join(results_dir, log_file), seed, True, selection_strategy))
                return learner.learn(domain, data, initial_indices)

            result = learn_bottom_up(data, learn_f, 3, 1)
        if result is None:
            flat[name][sample_count] = {"k": k, "h": h, "seed": seed, "bias": bias, "time_out": True}
        else:
            flat[name][sample_count] = {"k": k, "h": h, "seed": seed, "bias": bias, "time_out": False}
        if time_out is not None:
            flat[name][sample_count]["time_limit"] = time_out

    with open(overview, "w") as f:
        json.dump(flat, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("prefix")
    parser.add_argument("output_dir")
    parser.add_argument("bias")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-s", "--samples", default=None, type=int)
    parser.add_argument("-t", "--time_out", default=None, type=int)
    parser.add_argument("-a", "--non_incremental", default=False, action="store_true")
    parser.add_argument("-f", "--parameter_free", default=False, action="store_true")
    parsed = parser.parse_args()
    learn_synthetic(parsed.input_dir, parsed.prefix, parsed.output_dir, parsed.bias, parsed.plot, parsed.samples,
                    parsed.time_out, parsed.non_incremental, parsed.parameter_free)
