import argparse
import filecmp
import fnmatch
import json

import os
import shutil

import migrate


def combine(output_dir, dirs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    migrate.migrate_results(output_dir)

    summary = os.path.join(output_dir, "problems.txt")
    if not os.path.isfile(summary):
        flat = {}
    else:
        with open(summary, "r") as f:
            flat = json.load(f)

    for input_dir in dirs:
        migrate.migrate_results(input_dir)
        input_summary = os.path.join(input_dir, "problems.txt")
        with open(input_summary, "r") as f:
            input_flat = json.load(f)
        for problem_id in input_flat:
            if problem_id not in flat:
                flat[problem_id] = {}
            for sample_size in input_flat[problem_id]:
                if sample_size not in flat[problem_id]:
                    flat[problem_id][sample_size] = input_flat[problem_id][sample_size]
                else:
                    raise RuntimeError("Attempting to overwrite sample size {} for problem {} from file {}"
                                       .format(sample_size, problem_id, input_summary))

    for input_dir in dirs:
        for input_file in os.listdir('.'):
            if fnmatch.fnmatch(input_file, '*.learning_log.txt'):
                old_file = os.path.join(input_dir, input_file)
                new_file = os.path.join(output_dir, input_file)
                if not os.path.isfile(new_file):
                    shutil.copy(old_file, new_file)
                else:
                    if not filecmp.cmp(old_file, new_file):
                        raise RuntimeError("Attempting to overwrite {} with {}".format(new_file, old_file))

    with open(summary, "w") as f:
        json.dump(flat, f)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("dirs", nargs="*")
    parsed = parser.parse_args()
    combine(parsed.output_dir, parsed.dirs)


if __name__ == "__main__":
    parse()
