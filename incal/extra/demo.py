from __future__ import print_function, division

import argparse
import hashlib
import json
import random

import os
import tempfile

import time

import problem
import generator
import parse
import inc_logging

from os.path import basename

import pysmt.shortcuts as smt

from incremental_learner import RandomViolationsStrategy
from k_cnf_smt_learner import KCnfSmtLearner
from parameter_free_learner import learn_bottom_up


def learn(name, domain, h, data, seed):
    initial_size = 20
    violations_size = 10
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "results")
    problem_name = hashlib.sha256(name).hexdigest()

    def learn_inc(_data, _k, _h):
        violations_strategy = RandomViolationsStrategy(violations_size)
        learner = KCnfSmtLearner(_k, _h, violations_strategy)
        initial_indices = random.sample(list(range(len(data))), initial_size)
        log_file = os.path.join(log_dir, "{}_{}_{}.txt".format(problem_name, _k, _h))
        learner.add_observer(inc_logging.LoggingObserver(log_file, seed, True, violations_strategy))
        learned_theory = learner.learn(domain, data, initial_indices)
        # learned_theory = Or(*[And(*planes) for planes in hyperplane_dnf])
        print("Learned theory:\n{}".format(parse.smt_to_nested(learned_theory)))
        return learned_theory

    phi, k, h = learn_bottom_up(data, learn_inc, 1, 1, init_h=h, max_h=h)

    with open(os.path.join(log_dir, "problems.txt"), "a") as f:
        print(json.dumps({problem_name: name, "k": k, "h": h}), file=f)


def main(filename, sample_count):
    seed = time.time()
    random.seed(seed)

    target_formula = smt.read_smtlib(filename)

    variables = target_formula.get_free_variables()
    var_names = [str(v) for v in variables]
    var_types = {str(v): v.symbol_type() for v in variables}
    var_domains = {str(v): (0, 200) for v in variables}  # TODO This is a hack

    domain = problem.Domain(var_names, var_types, var_domains)
    name = basename(filename).split(".")[0]
    target_problem = problem.Problem(domain, target_formula, name)

    # compute_difference(domain, target_formula, target_formula)

    samples = generator.get_problem_samples(target_problem, sample_count, 1)

    initial_indices = random.sample(list(range(sample_count)), 20)
    learner = KCnfSmtLearner(3, 3, RandomViolationsStrategy(5))

    dir_name = "../output/{}".format(name)
    img_name = "{}_{}_{}".format(learner.name, sample_count, seed)
    # learner.add_observer(plotting.PlottingObserver(data_set.samples, dir_name, img_name, "r0", "r1"))
    with open("log.txt", "w") as f:
        learner.add_observer(inc_logging.LoggingObserver(f))

        print(parse.smt_to_nested(learner.learn(domain, samples, initial_indices)))


def compute_difference(domain, target_theory, learned_theory):
    query = (target_theory & ~learned_theory) | (~target_theory & learned_theory)
    compute_wmi(domain, query, domain.variables)


def compute_wmi(domain, query, variables):
    # os.environ["PATH"] += os.pathsep + "/Users/samuelkolb/Downloads/latte/dest/bin"
    # from sys import path
    # path.insert(0, "/Users/samuelkolb/Documents/PhD/wmi-pa/src")
    # from wmi import WMI

    # support = []
    # for v in domain.real_vars:
    #     lb, ub = domain.var_domains[v]
    #     sym = domain.get_symbol(v)
    #     support.append((lb <= sym) & (sym <= ub))
    #
    # support = smt.And(*support)
    # wmi = WMI()
    # total_volume, _ = wmi.compute(support, 1, WMI.MODE_PA)
    # query_volume, _ = wmi.compute(support & query, 1, WMI.MODE_PA)
    # print(query_volume / total_volume)

    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        flat = {
            "domain": problem.export_domain(domain, to_str=False),
            "query": parse.smt_to_nested(query),
            "variables": variables
        }
        json.dump(flat, f)
        with open("test.txt", "w") as f2:
            json.dump(flat, f2)
        f.close()
    finally:
        os.remove(f.name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("sample_count", type=int)
    args = parser.parse_args()
    main(args.filename, args.sample_count)
