import argparse
import os
import shutil
import urllib.request


def get_res_root(*args):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "res", *args)


def prepare_smt_lib_benchmark():
    qf_lra_folder = get_res_root("smt_lib_benchmark")
    if not os.path.exists(qf_lra_folder):
        os.makedirs(qf_lra_folder)
        url = "http://smt-lib.loria.fr/zip/QF_LRA.zip"
        with urllib.request.urlopen(url) as response, open(os.path.join(qf_lra_folder, "qf_lra.zip"), 'wb') as out_file:
            shutil.copyfileobj(response, out_file)



def prepare_synthetic():
    pass


def main():
    smt_lib_name = "smt-lib-benchmark"
    synthetic_name = "synthetic"
    parser = argparse.ArgumentParser(description="Prepare data for experiments")
    subparsers = parser.add_subparsers(dest="source")
    benchmark_parser = subparsers.add_parser(smt_lib_name)
    synthetic_parser = subparsers.add_parser(synthetic_name)

    args = parser.parse_args()
    if args.source == smt_lib_name:
        prepare_smt_lib_benchmark()
    elif args.source == synthetic_name:
        prepare_synthetic()


if __name__ == "__main__":
    main()
