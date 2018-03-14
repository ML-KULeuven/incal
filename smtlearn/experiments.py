from __future__ import print_function

import argparse
import json
import random

import os

import time

from generator import import_synthetic_data_files
from inc_logging import LoggingObserver
from incremental_learner import AllViolationsStrategy, RandomViolationsStrategy, WeightedRandomViolationsStrategy
from k_cnf_smt_learner import KCnfSmtLearner
from k_dnf_smt_learner import KDnfSmtLearner
from parameter_free_learner import learn_bottom_up
from timeout import timeout


class IncrementalConfig(object):
    def __init__(self, initial, initial_size, selection, selection_size):
        self.initial = initial
        self.initial_size = initial_size
        self.selection = selection
        self.selection_size = selection_size
        self.domain = None
        self.data = None
        self.dt_weights = None

    def set_data(self, data):
        self.data = data
        self.dt_weights = None

    def get_dt_weights(self):
        if self.dt_weights is None:
            import dt_selection
            self.dt_weights = [min(d.values()) for d in dt_selection.get_distances(self.domain, self.data)]
        return self.dt_weights

    def get_initial_indices(self):
        if self.initial is None:
            return list(range(len(self.data)))
        elif self.initial == "random":
            return random.sample(range(len(self.data)), self.initial_size)
        elif self.initial == "dt_weighted":
            import sampling
            return sampling.sample_weighted(zip(range(len(self.data)), self.get_dt_weights()), self.initial_size)
        else:
            raise RuntimeError("Unknown initial type {}".format(self.initial))

    def get_selection_strategy(self):
        if self.selection is None:
            return RandomViolationsStrategy(0)
        elif self.selection == "random":
            return RandomViolationsStrategy(self.selection_size)
        elif self.selection == "dt_weighted":
            return WeightedRandomViolationsStrategy(self.selection_size, self.get_dt_weights())
        else:
            raise RuntimeError("Unknown selection type {}".format(self.selection))


def learn_synthetic(input_dir, prefix, results_dir, bias, incremental_config, plot=None, sample_count=None,
                    time_out=None, parameter_free=False):

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
        domain = synthetic_problem.theory_problem.domain

        if name not in flat:
            flat[name] = {}

        print(name)

        seed = hash(time.time())
        random.seed(seed)

        if sample_count is not None and sample_count < len(data):
            data = random.sample(data, sample_count)
        else:
            sample_count = len(data)

        incremental_config.set_data(data)

        if not parameter_free:
            initial_indices = incremental_config.get_initial_indices()
            h = synthetic_problem.half_space_count
            k = synthetic_problem.formula_count

            if bias == "cnf" or bias == "dnf":
                selection_strategy = incremental_config.get_selection_strategy()
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
                selection_strategy = incremental_config.get_selection_strategy()
                if bias == "cnf":
                    learner = KCnfSmtLearner(_k, _h, selection_strategy)
                elif bias == "dnf":
                    learner = KDnfSmtLearner(_k, _h, selection_strategy)
                initial_indices = incremental_config.get_initial_indices()
                log_file = "{}_{}_{}_{}_{}.learning_log.txt".format(name, sample_count, seed, _k, _h)
                learner.add_observer(LoggingObserver(os.path.join(results_dir, log_file), seed, True, selection_strategy))
                return learner.learn(domain, data, initial_indices)

            result, k, h = learn_bottom_up(data, learn_f, 3, 1)
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
    parser.add_argument("--bias", default="cnf")
    parser.add_argument("--initial", default="random")
    parser.add_argument("--initial_size", default=20, type=int)
    parser.add_argument("--selection", default="random")
    parser.add_argument("--selection_size", default=10, type=int)
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-s", "--samples", default=None, type=int)
    parser.add_argument("-t", "--time_out", default=None, type=int)
    parser.add_argument("-a", "--non_incremental", default=False, action="store_true")
    parser.add_argument("-f", "--parameter_free", default=False, action="store_true")
    parsed = parser.parse_args()

    if parsed.non_incremental:
        inc_config = IncrementalConfig(None, None, None, None)
    else:
        inc_config = IncrementalConfig(parsed.initial, parsed.initial_size, parsed.selection, parsed.selection_size)

    learn_synthetic(parsed.input_dir, parsed.prefix, parsed.output_dir, parsed.bias, inc_config,
                    parsed.plot, parsed.samples, parsed.time_out, parsed.parameter_free)
