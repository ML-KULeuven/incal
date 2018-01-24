import argparse
import glob
import json

import os
import shutil

import re


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


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", help="Specify the type of migration")
    parser.add_argument("-r", "--results", help="Specify the result directory")
    parser.add_argument("-b", "--bias", default=None, help="Specify the bias")
    parsed = parser.parse_args()
    if parsed.type == "results":
        migrate_results(parsed.results, parsed.bias)


if __name__ == "__main__":
    parse()
