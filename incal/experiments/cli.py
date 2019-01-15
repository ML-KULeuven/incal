import argparse

import numpy as np
from pywmi.smt_print import pretty_print

from .learn import learn_benchmark
from .prepare import prepare_smt_lib_benchmark, prepare_ratios, prepare_samples, prepare_synthetic
from incal.learn import LearnOptions
from . import examples


def main():
    smt_lib_name = "smt-lib-benchmark"
    synthetic_name = "synthetic"
    parser = argparse.ArgumentParser(description="Interface with benchmark or synthetic data for experiments")

    parser.add_argument("source")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)

    task_parsers = parser.add_subparsers(dest="task")
    prepare_parser = task_parsers.add_parser("prepare")
    prepare_parser.add_argument("--reset_samples", type=bool, default=False)
    learn_parser = task_parsers.add_parser("learn")

    learn_options = LearnOptions()
    learn_options.add_arguments(learn_parser)

    args = parser.parse_args()
    if args.task == "prepare":
        if args.source == smt_lib_name:
            prepare_smt_lib_benchmark()
            prepare_ratios()
            prepare_samples(args.runs, args.sample_size, args.reset_samples)
        elif args.source == synthetic_name:
            prepare_synthetic()
    elif args.task == "learn":
        learn_options.parse_arguments(args)
        if args.source == smt_lib_name:
            learn_benchmark(args.runs, args.sample_size, learn_options)
        elif args.source == synthetic_name:
            pass
        elif args.source.startswith("ex"):
            example_name = args.source.split(":", 1)[1]
            domain, formula = examples.get_by_name(example_name)
            np.random.seed(1)
            from pywmi.sample import uniform
            samples = uniform(domain, args.sample_size)
            from pywmi import evaluate
            labels = evaluate(domain, formula, samples)
            learn_options.set_value("domain", domain, False)
            learn_options.set_value("data", samples, False)
            learn_options.set_value("labels", labels, False)
            (formula, k, h), duration = learn_options.call(True)
            print("[{:.2f}s] Learned formula (k={}, h={}): {}".format(duration, k, h, pretty_print(formula)))


if __name__ == "__main__":
    main()
