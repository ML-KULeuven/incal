import argparse
import json

import os
import shutil

import re

import parse
import problem
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
    print(compute_wmi(domain, [smt.Iff(target_formula, smt.Not(learned_formula))]))


def add_accuracy(results_dir, data_dir=None):
    results_flat = load_results(results_dir)

    for problem_id in results_flat:
        for sample_size in results_flat[problem_id]:
            config = results_flat[problem_id][sample_size]
            timed_out = config.get("time_out", False)
            if not timed_out:
                learned_formula = None
                for message in get_log_messages(results_dir, config):
                    if message["type"] == "update":
                        learned_formula = parse.nested_to_smt(message["theory"])

                if data_dir is not None:
                    with open(os.path.join(data_dir, "{}.txt".format(str(config["problem_id"])))) as f:
                        import generator
                        s_problem = generator.import_synthetic_data(json.load(f))
                    target_formula = s_problem.synthetic_problem.theory_problem.theory
                    domain = s_problem.synthetic_problem.theory_problem.domain
                else:
                    raise RuntimeError("Not yet implemented")
                config["accuracy"] = calculate_accuracy(domain, target_formula, learned_formula)

    dump_results(results_flat, results_dir)

if __name__ == "__main__":
    x = smt.Symbol("x", smt.REAL)
    calculate_accuracy(problem.Domain(["x"], {"x": smt.REAL}, {"x": (0, 1)}), x <= smt.Real(0.5), x <= smt.Real(0.4))