import argparse

import os

import math


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="mode")

        scan_parser = subparsers.add_parser("scan", help="Scan the directory and load smt problems")
        scan_parser.add_argument("-d", "--dir", default=None, help="Specify the directory to load files from")

        learn_parser = subparsers.add_parser("learn", help="Learn SMT formulas")
        learn_parser.add_argument("dir", help="Specify the results directory")
        learn_parser.add_argument("-s", "--samples", type=int, help="Specify the number of samples for learning")
        learn_parser.add_argument("-a", "--all", default=False, action="store_true",
                                  help="If set, learning will not use incremental mode and include all examples")
        learn_parser.add_argument("-d", "--dnf", default=False, action="store_true",
                                  help="If set, learning bias is DNF instead of CNF")

        table_parser = subparsers.add_parser("table", help="Types can be: [time, k, h, id, acc, samples, l]")
        table_parser.add_argument("row_key", help="Specify the row key type")
        table_parser.add_argument("col_key", default=None, help="Specify the col key type")
        table_parser.add_argument("value", default=None, help="Specify the value type")
        table_parser.add_argument("dirs", nargs="*", help="Specify the directories to load files from, always in pairs:"
                                                          "result_dir, data_dir")

        table_subparsers = table_parser.add_subparsers(dest="command")
        table_print_parser = table_subparsers.add_parser("print", help="Print the table")
        table_print_parser.add_argument("-d", "--delimiter", default="\t", help="Specify the delimiter (default=tab)")
        table_print_parser.add_argument("-a", "--aggregate", default=False, action="store_true",
                                       help="Aggregate the rows in the plot")

        table_plot_parser = table_subparsers.add_parser("plot", help="Plot the table")
        table_plot_parser.add_argument("-a", "--aggregate", default=False, action="store_true",
                                       help="Aggregate the rows in the plot")
        table_plot_parser.add_argument("--y_min", default=None, type=float, help="Minimum value for y")
        table_plot_parser.add_argument("--y_max", default=None, type=float, help="Maximum value for y")
        table_plot_parser.add_argument("--x_min", default=None, type=float, help="Minimum value for x")
        table_plot_parser.add_argument("--x_max", default=None, type=float, help="Maximum value for x")
        table_plot_parser.add_argument("--legend_pos", default=None, type=str, help="Legend position")
        table_plot_parser.add_argument("-o", "--output", default=None, help="Specify the output file")

        combine_parser = subparsers.add_parser("combine", help="Combine multiple results directories")
        combine_parser.add_argument("output_dir", help="The output directory to summarize results in")
        combine_parser.add_argument("input_dirs", nargs="*", help="Specify the directories to combine")
        combine_parser.add_argument("-b", "--bias", default=None, help="Specify the bias")
        combine_parser.add_argument("-p", "--prefix", default=None, help="Specify the prefix for input dirs")

        gen_parser = subparsers.add_parser("generate", help="Generate synthetic examples")
        gen_parser.add_argument("data_dir")
        gen_parser.add_argument("-n", "--data_sets", default=10, type=int)
        gen_parser.add_argument("--prefix", default="synthetics")
        gen_parser.add_argument("-b", "--bool_count", default=2, type=int)
        gen_parser.add_argument("-r", "--real_count", default=2, type=int)
        gen_parser.add_argument("--bias", default="cnf")
        gen_parser.add_argument("-k", "--k", default=3, type=int)
        gen_parser.add_argument("-l", "--literals", default=4, type=int)
        gen_parser.add_argument("--half_spaces", default=7, type=int)
        gen_parser.add_argument("-s", "--samples", default=1000, type=int)
        gen_parser.add_argument("--ratio", default=90, type=int)
        gen_parser.add_argument("-p", "--plot_dir", default=None)
        gen_parser.add_argument("-e", "--errors", default=0, type=int)

        migration_parser = subparsers.add_parser("migrate", help="Migrate files to newer or extended versions")
        migration_subparsers = migration_parser.add_subparsers(dest="type")

        migration_fix_parser = migration_subparsers.add_parser("fix", help="Fix result files")
        migration_fix_parser.add_argument("results_dir", help="Specify the result directory")
        migration_fix_parser.add_argument("-b", "--bias", default=None, help="Specify the bias")

        migration_acc_parser = migration_subparsers.add_parser("accuracy", help="Add accuracy to result files")
        migration_acc_parser.add_argument("results_dir", help="Specify the result directory")
        migration_acc_parser.add_argument("-d", "--data_dir", help="Specify the data directory for synthetic problems")
        migration_acc_parser.add_argument("-s", "--samples", default=None, help="Specify the number of samples", type=int)
        migration_acc_parser.add_argument("-f", "--force", default=False, action="store_true", help="Overwrites existing values")

        migration_ratio_parser = migration_subparsers.add_parser("ratio", help="Add ratio to result files")
        migration_ratio_parser.add_argument("results_dir", help="Specify the result directory")
        migration_ratio_parser.add_argument("-d", "--data_dir", help="Specify the data directory for synthetic problems")
        migration_ratio_parser.add_argument("-s", "--samples", default=None, help="Specify the number of samples", type=int)
        migration_ratio_parser.add_argument("-f", "--force", default=False, action="store_true", help="Overwrites existing values")

        args = parser.parse_args()

        if args.mode == "scan":
            full_dir = os.path.abspath(args.filename)
            root_dir = os.path.dirname(full_dir)

            import smt_scan
            smt_scan.scan(full_dir, root_dir)
            smt_scan.analyze(root_dir)
            smt_scan.ratios()
        elif args.mode == "learn":
            import smt_scan
            smt_scan.learn(args.samples, args.dir, args.all, args.dnf)
        elif args.mode == "table":
            import smt_scan
            table = smt_scan.TableMaker(args.row_key, args.col_key, args.value)
            for i in range(int(math.floor(len(args.dirs) / 3))):
                table.load_table(args.dirs[3 * i], args.dirs[3 * i + 1], args.dirs[3 * i + 2])
            if args.command == "print":
                table.delimiter = args.delimiter
                print(table.to_txt(0, args.aggregate))
            elif args.command == "plot":
                table.plot_table(args.output, None if args.aggregate else 0, args.y_min, args.y_max, args.x_min, args.x_max, args.legend_pos)
            else:
                print("Error: unknown table command {}".format(args.command))
        elif args.mode == "combine":
            import combine_results
            combine_results.combine(args.output_dir, args.input_dirs, args.bias, args.prefix)
        elif args.mode == "generate":
            from generator import generate_random
            generate_random(args.data_sets, args.prefix, args.bool_count, args.real_count, args.bias, args.k,
                            args.literals, args.half_spaces, args.samples, args.ratio, args.errors, args.data_dir,
                            args.plot_dir)
        elif args.mode == "migrate":
            import migrate
            if args.type == "fix":
                migrate.migrate_results(args.results_dir, args.bias)
            elif args.type == "accuracy":
                migrate.add_accuracy(args.results_dir, args.data_dir, args.samples, args.force)
            elif args.type == "ratio":
                migrate.add_ratio(args.results_dir, args.data_dir, args.samples, args.force)
            else:
                print("Error: unknown migration type {}".format(args.type))
        else:
            print("Error: unknown mode {}".format(args.mode))


    parse_args()
