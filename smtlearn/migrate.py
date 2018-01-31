from __future__ import print_function, division

import argparse
import json

import os
import random
import shutil

import re

import time
from bitarray import bitarray

import generator
import parse
import problem
from smt_check import test
from smt_print import pretty_print
from smt_scan import load_results, get_log_messages, dump_results
import pysmt.shortcuts as smt


def migrate_results(directory, bias=None):
    summary = os.path.join(directory, "problems.txt")
    if os.path.isfile(summary):
        with open(summary, "r") as f:
            flat = json.load(f)

        for problem_id in flat:
            for sample_size in flat[problem_id]:
                if "bias" not in flat[problem_id][sample_size]:
                    flat[problem_id][sample_size]["bias"] = "cnf" if bias is None else bias

                seed, k, h = (flat[problem_id][sample_size][v] for v in ["seed", "k", "h"])

                pattern = r'{problem_id}_{size}_{seed}_\d+_\d+.txt' \
                    .format(problem_id=problem_id, size=sample_size, seed=seed)
                for old_file in os.listdir(directory):
                    if re.match(pattern, old_file):
                        new_file = old_file[:-4] + ".learning_log.txt"
                        shutil.move(os.path.join(directory, old_file), os.path.join(directory, new_file))

        with open(summary, "w") as f:
            json.dump(flat, f)


def calculate_accuracy(domain, target_formula, learned_formula):
    from sys import path
    path.insert(0, "/Users/samuelkolb/Documents/PhD/wmi-pa/experiments/client")
    from run import compute_wmi
    print("Calculate accuracy:")
    # print(pretty_print(target_formula))
    # print(pretty_print(learned_formula))

    # r0, r1 = [smt.Symbol(n, smt.REAL) for n in ["r0", "r1"]]
    # b0, b1, b2, b3 = [smt.Symbol(n, smt.BOOL) for n in ["b0", "b1", "b2", "b3"]]
    # t1 = (~(1.0 <= 0.427230115861 * r0 + 1.02084935803 * r1) | ~(1.0 <= 1.59402729715 * r0 + 0.309004054118 * r1) | ~b1)
    # t2 = (b2 | (1.0 <= 1.59402729715 * r0 + 0.309004054118 * r1) | ~b0)

    # domain = problem.Domain(["x", "y"], {"x": smt.REAL, "y": smt.REAL}, {"x": (0, 1), "y": (0, 1)})
    # x, y = smt.Symbol("x", smt.REAL), smt.Symbol("y", smt.REAL)
    # t2 = (1.0 <= 1.5 * x + 0.5 * y)
    # t2 = (2 <= 3 * x + y)
    # f = (t1 & t2)

    print(domain)
    print(pretty_print(target_formula))
    print(pretty_print(learned_formula))
    accuracy = list(compute_wmi(domain, [smt.Iff(target_formula, learned_formula)]))[0]
    print(accuracy)
    return accuracy


def calculate_accuracy_approx(domain, target_formula, learned_formula, samples):
    bits_target = bitarray([test(target_formula, sample) for sample in samples])
    bits_learned = bitarray([test(learned_formula, sample) for sample in samples])
    accuracy = ((bits_target & bits_learned) | (~bits_target & ~bits_learned)).count() / len(samples)
    print(accuracy)
    return accuracy


def adapt_domain_multiple(target_problem, new_bounds):
    domain = target_problem.domain
    adapted_domain = problem.Domain(domain.variables, domain.var_types, new_bounds)
    return problem.Problem(adapted_domain, target_problem.theory, target_problem.name)


def get_problem(data_dir, problem_id):
    try:
        with open(os.path.join(data_dir, "{}.txt".format(str(problem_id)))) as f:
            import generator
            s_problem = generator.import_synthetic_data(json.load(f))
        return s_problem.synthetic_problem.theory_problem
    except IOError:
        with open(os.path.join(data_dir, "problems", "{}.txt".format(str(problem_id)))) as f:
            import generator
            theory_problem = problem.import_problem(json.load(f))

        with open(os.path.join(data_dir, "summary.json"), "r") as f:
            flat = json.load(f)
        ratio_dict = flat["ratios"]
        lookup = flat["lookup"]

        adapted_problem = adapt_domain_multiple(theory_problem, ratio_dict[lookup[problem_id]]["bounds"])

        return adapted_problem


def add_accuracy(results_dir, data_dir=None, acc_sample_size=None, recompute=False):
    results_flat = load_results(results_dir)

    for problem_id in results_flat:

        if data_dir is not None:
            theory_problem = get_problem(data_dir, problem_id)
            domain = theory_problem.domain
            target_formula = theory_problem.theory
            print(problem_id)
            print(pretty_print(target_formula))
        else:
            raise RuntimeError("Data directory missing")

        for sample_size in results_flat[problem_id]:
            config = results_flat[problem_id][sample_size]
            timed_out = config.get("time_out", False)
            if not timed_out:
                learned_formula = None
                for message in get_log_messages(results_dir, config, p_id=problem_id, samples=sample_size):
                    if message["type"] == "update":
                        learned_formula = parse.nested_to_smt(message["theory"])

                print(pretty_print(learned_formula))
                print()

                if acc_sample_size is None:
                    if recompute or "exact_accuracy" not in config:
                        config["exact_accuracy"] = calculate_accuracy(domain, target_formula, learned_formula)
                else:
                    if recompute or "approx_accuracy" not in config:
                        config["approx_accuracy"] = dict()
                    acc_dict = config["approx_accuracy"]
                    if acc_sample_size not in acc_dict:
                        acc_dict[acc_sample_size] = []
                    if len(acc_dict[acc_sample_size]) < 1:
                        seed = hash(time.time())
                        random.seed(seed)
                        samples = [generator.get_sample(domain) for _ in range(acc_sample_size)]
                        acc_dict[acc_sample_size].append({
                            "acc": calculate_accuracy_approx(domain, target_formula, learned_formula, samples),
                            "seed": seed,
                        })

    dump_results(results_flat, results_dir)


def calculate_ratio(domain, formula):
    raise NotImplementedError()


def calculate_ratio_approx(formula, samples):
    bits = bitarray([test(formula, sample) for sample in samples])
    positives = bits.count() / len(samples)
    ratio = max(positives, 1 - positives)
    print("Ratio: {}".format(ratio))
    return ratio


def add_ratio(results_dir, data_dir=None, ratio_sample_size=None, recompute=False):
    results_flat = load_results(results_dir)

    ratio_cache = dict()

    for problem_id in results_flat:
        if data_dir is not None:
            theory_problem = get_problem(data_dir, problem_id)
            domain = theory_problem.domain
            formula = theory_problem.theory
        else:
            raise RuntimeError("Data directory missing")

        seed = hash(time.time())
        random.seed(seed)
        samples = [generator.get_sample(domain) for _ in range(ratio_sample_size)]

        ratio = calculate_ratio(domain, formula) if ratio_sample_size is None else calculate_ratio_approx(formula, samples)

        for sample_size in results_flat[problem_id]:
            config = results_flat[problem_id][sample_size]

            if ratio_sample_size is None:
                if recompute or "exact_ratio" not in config:
                    config["exact_ratio"] = ratio
            else:
                if recompute or "approx_ratio" not in config:
                    config["approx_ratio"] = dict()
                ratio_dict = config["approx_ratio"]
                if ratio_sample_size not in ratio_dict:
                    ratio_dict[ratio_sample_size] = []
                if len(ratio_dict[ratio_sample_size]) < 1:
                    ratio_dict[ratio_sample_size].append({
                        "ratio": ratio,
                        "seed": seed,
                    })

    dump_results(results_flat, results_dir)


if __name__ == "__main__":
    x = smt.Symbol("x", smt.REAL)
    calculate_accuracy(problem.Domain(["x"], {"x": smt.REAL}, {"x": (0, 1)}), x <= smt.Real(0.5), x <= smt.Real(0.4))