import argparse

import numpy as np
from pywmi.smt_print import pretty_print

from .learn import learn_benchmark, get_experiment, learn_synthetic
from .prepare import prepare_smt_lib_benchmark, prepare_ratios, prepare_samples, prepare_synthetic
from incal.learn import LearnOptions
from . import examples
from .analyze import analyze
from incal.util import analyze as show


def main():
    smt_lib_name = "smt-lib-benchmark"
    synthetic_name = "synthetic"
    parser = argparse.ArgumentParser(description="Interface with benchmark or synthetic data for experiments")

    parser.add_argument("source")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--time_out", type=int, default=None)

    task_parsers = parser.add_subparsers(dest="task")
    prepare_parser = task_parsers.add_parser("prepare")
    prepare_parser.add_argument("--reset_samples", type=bool, default=False)
    learn_parser = task_parsers.add_parser("learn")
    analyze_parser = task_parsers.add_parser("analyze")
    analyze_parser.add_argument("--dirs", nargs="+", type=str)
    analyze_parser.add_argument("--res_path", type=str, default=None)

    show_parsers = analyze_parser.add_subparsers()
    show_parser = show_parsers.add_parser("show")
    show.add_arguments(show_parser)

    learn_options = LearnOptions()
    learn_options.add_arguments(learn_parser)

    args = parser.parse_args()
    if args.task == "prepare":
        if args.source == smt_lib_name:
            prepare_smt_lib_benchmark()
            prepare_ratios()
            prepare_samples(args.runs, args.sample_size, args.reset_samples)
        elif args.source == synthetic_name:
            prepare_synthetic(args.input_dir, args.output_dir, args.runs, args.sample_size)
    elif args.task == "learn":
        learn_options.parse_arguments(args)
        if args.source == smt_lib_name:
            learn_benchmark(args.runs, args.sample_size, args.processes, args.time_out, learn_options)
        elif args.source == synthetic_name:
            learn_synthetic(args.input_dir, args.output_dir, args.runs, args.sample_size, args.processes, args.time_out, learn_options)
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
    elif args.task == "analyze":
        analyze(args.dirs, args.res_path, show.parse_args(args))



if __name__ == "__main__":
    main()
