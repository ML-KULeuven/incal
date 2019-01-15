import argparse

from .learn import learn_benchmark
from .prepare import prepare_smt_lib_benchmark, prepare_ratios, prepare_samples, prepare_synthetic
from incal.learn import LearnOptions


def main():
    smt_lib_name = "smt-lib-benchmark"
    synthetic_name = "synthetic"
    parser = argparse.ArgumentParser(description="Interface with benchmark or synthetic data for experiments")

    parser.add_argument("source")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)

    task_parsers = parser.add_subparsers(dest="task")
    prepare_parser = task_parsers.add_parser("prepare")
    learn_parser = task_parsers.add_parser("learn")

    learn_options = LearnOptions()
    learn_options.add_arguments(learn_parser)

    args = parser.parse_args()
    if args.task == "prepare":
        if args.source == smt_lib_name:
            prepare_smt_lib_benchmark()
            prepare_ratios()
            prepare_samples(100, 10000)
        elif args.source == synthetic_name:
            prepare_synthetic()
    elif args.task == "learn":
        if args.source == smt_lib_name:
            learn_options.parse_arguments(args)
            learn_benchmark(args.runs, args.sample_size, learn_options)
        elif args.source == synthetic_name:
            pass


if __name__ == "__main__":
    main()
