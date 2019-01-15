import argparse
import glob
import hashlib
import itertools
import os
import pickle
import random
import shutil
import time
import traceback
import urllib.request
import zipfile
from collections import defaultdict
from typing import Tuple

import numpy as np
import pysmt.shortcuts as smt
import pysmt.environment
from pywmi import Domain, smt_to_nested, evaluate, RejectionEngine
from pywmi.domain import Density
from pywmi.sample import uniform

from .find_hyperplanes import HalfSpaceWalker
from .find_operators import OperatorWalker


def get_res_root(*args):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "res", *args)


def get_benchmark_dir():
    return get_res_root("smt_lib_benchmark")


def get_benchmark_cache_dir():
    return os.path.join(get_benchmark_dir(), "qf_lra_cache")


def get_benchmark_samples_dir():
    return os.path.join(get_benchmark_dir(), "qf_lra_samples")


def get_benchmark_results_dir():
    return os.path.join(get_benchmark_dir(), "qf_lra_results")


def get_summary_file():
    return os.path.join(get_benchmark_dir(), "qf_lra_summary.pickle")


# https://stackoverflow.com/a/11385480/253387
def fix_zip_file(zip_file):
    with open(zip_file, 'r+b') as f:
        data = f.read()
        pos = data.find(b'\x50\x4b\x05\x06')  # End of central directory signature
        if pos > 0:
            f.seek(pos + 22)  # size of 'ZIP end of central directory record'
            f.truncate()
        else:
            pass


def checksum(filename):
    hash_engine = hashlib.sha512()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_engine.update(chunk)
    return hash_engine.hexdigest()


def import_problem(filename):
    target_formula = smt.read_smtlib(filename)

    variables = target_formula.get_free_variables()
    var_names = [str(v) for v in variables]
    var_types = {str(v): v.symbol_type() for v in variables}
    var_domains = {str(v): (None, None) for v in variables}  # TODO This is a hack

    return (Domain(var_names, var_types, var_domains)), target_formula


def prepare_smt_lib_benchmark():
    benchmark_folder = get_res_root("smt_lib_benchmark")
    if not os.path.exists(benchmark_folder):
        os.makedirs(benchmark_folder)

    zip_file = os.path.join(benchmark_folder, "qf_lra.zip")
    zip_checksums = [
        'd0031a9e1799f78e72951aa4bacedaff7c0d027905e5de29b5980083b9c51138def165cc18fff205c1cdd0ef60d5d95cf179f0d82ec41ba489acf4383f3e783c']  # ,
    # '8aa31ada44bbb6705ce58f1f50870da4f3b2d2d27065f3c5c6a17bd484a4cb7eab0c1d55a8d78e48217e66c5b2d876c0708516fb8a383d1ea82a6d4f1278d476']
    qf_lra_folder = os.path.join(benchmark_folder, "QF_LRA")
    if not os.path.exists(qf_lra_folder) and not os.path.exists(zip_file):
        print("Downloading ZIP file to {}".format(zip_file))
        url = "http://smt-lib.loria.fr/zip/QF_LRA.zip"
        with urllib.request.urlopen(url) as response, open(zip_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    if not os.path.exists(qf_lra_folder):
        print("Extracting ZIP file {}".format(zip_file))
        if checksum(zip_file) not in zip_checksums:
            fix_zip_file(zip_file)
            if checksum(zip_file) not in zip_checksums:
                raise RuntimeError("Corrupted file download")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(benchmark_folder)

    summary_file = os.path.join(benchmark_folder, "qf_lra_summary.pickle")

    if not os.path.exists(summary_file):
        with open(summary_file, "wb") as summary_file_ref:
            pickle.dump(dict(), summary_file_ref)
    with open(summary_file, "rb") as summary_file_ref:
        summary = pickle.load(summary_file_ref)

    cache_dir = os.path.join(benchmark_folder, "qf_lra_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    for filename in glob.glob("{}/**/*.smt*".format(qf_lra_folder), recursive=True):
        name = filename[filename.find("QF_LRA"):]
        if name not in summary:
            summary[name] = dict()

        cache_filename = "{}{}{}".format(cache_dir, os.path.sep, name)

        entry = summary[name]
        if "file_size" not in entry:
            entry["file_size"] = os.path.getsize(filename)

        if entry["file_size"] / 1024 > 100:
            continue

        density_filename = "{}.density".format(cache_filename)
        if not os.path.exists(os.path.dirname(density_filename)):
            os.makedirs(os.path.dirname(density_filename))

        domain, formula = None, None

        pysmt.environment.push_env()
        pysmt.environment.get_env().enable_infix_notation = True

        if not os.path.exists(density_filename):
            print("Importing {}".format(name))
            try:
                domain, formula = import_problem(filename)
                Density(domain, formula, smt.Real(1.0)).export_to(density_filename)
            except RuntimeError:
                print("Error")
                continue

        keys = ["real_variables_count", "bool_variables_count", "operators", "half_spaces"]
        if any(k not in entry for k in keys) and (domain is None or formula is None):
            print("Loading {}".format(name))
            density = Density.import_from(density_filename)
            domain = density.domain
            formula = density.support

        if "real_variables_count" not in entry:
            entry["real_variables_count"] = len(domain.real_vars)
        if "bool_variables_count" not in entry:
            entry["bool_variables_count"] = len(domain.bool_vars)
        if "operators" not in entry:
            entry["operators"] = OperatorWalker().find_operators(formula)
        if "half_spaces" not in entry:
            entry["half_spaces"] = HalfSpaceWalker().find_half_spaces(formula)

        pysmt.environment.pop_env()

    with open(summary_file, "wb") as summary_file_ref:
        pickle.dump(summary, summary_file_ref)


def select_benchmark_files(entry_filter):
    summary_file = get_summary_file()

    with open(summary_file, "rb") as summary_file_ref:
        summary = pickle.load(summary_file_ref)

    cache_dir = get_benchmark_cache_dir()
    for name, entry in summary.items():
        if entry_filter(entry):
            cache_filename = "{}{}{}".format(cache_dir, os.path.sep, name)
            density_filename = "{}.density".format(cache_filename)

            yield name, entry, density_filename


def benchmark_filter(entry):
    return "real_variables_count" in entry and entry["real_variables_count"] + entry["bool_variables_count"] <= 10 and \
           "=" not in entry["operators"]


def edit_summary(callback):
    with open(get_summary_file(), "rb") as summary_file_reference:
        summary = pickle.load(summary_file_reference)

    callback(summary)

    with open(get_summary_file(), "wb") as summary_file_reference:
        pickle.dump(summary, summary_file_reference)


def prepare_ratios():
    sample_count = 1000
    bounds_pool = [(-1, 1), (-10, 10), (-100, 100), (-1000, 1000)]
    ratios = dict()
    for name, entry, density_filename in select_benchmark_files(lambda e: "bounds" not in e and benchmark_filter(e)):
        print("Finding ratios for {}".format(name))
        pysmt.environment.push_env()
        pysmt.environment.get_env().enable_infix_notation = True

        density = Density.import_from(density_filename)
        domain = density.domain

        result_bounds = []
        result_ratios = []
        for bounds in itertools.product(*[bounds_pool for _ in range(len(domain.real_vars))]):
            var_bounds = dict(zip(domain.real_vars, bounds))
            restricted_domain = Domain(domain.variables, domain.var_types, var_bounds)
            samples = uniform(restricted_domain, sample_count)
            labels = evaluate(restricted_domain, density.support, samples)
            positive_count = sum(labels)
            if 0 < positive_count < sample_count:
                ratio = positive_count / sample_count
                result_bounds.append(var_bounds)
                result_ratios.append(ratio)

        ratios[name] = list(zip(result_bounds, result_ratios))
        print(name, result_ratios)

        pysmt.environment.pop_env()

    with open(get_summary_file(), "rb") as summary_file_reference:
        summary = pickle.load(summary_file_reference)

    for name, bounds in ratios.items():
        summary[name]["bounds"] = bounds

    with open(get_summary_file(), "wb") as summary_file_reference:
        pickle.dump(summary, summary_file_reference)


def prepare_samples(n, sample_size, reset):
    samples_dir = get_benchmark_samples_dir()

    seeds = [random.randint(0, 2 ** 32 - 1) for _ in range(n)]
    samples_dict = dict()

    def sample_filter(_entry):
        if "bounds" in _entry and benchmark_filter(_entry):
            if "samples" not in _entry["samples"]:
                return True
            else:
                return reset or any(
                    len([
                        s for s in _entry["samples"]
                        if s["sample_size"] == sample_size and s["bounds"] == _bounds[0]]
                    ) < n
                    for _bounds in _entry["bounds"] if 0.2 <= _bounds[1] <= 0.8
                )
        return False

    for name, entry, filename in select_benchmark_files(sample_filter):
        print("Creating samples for {}".format(name))
        pysmt.environment.push_env()
        pysmt.environment.get_env().enable_infix_notation = True

        density = Density.import_from(filename)
        samples_dict[name] = [] if reset else entry.get("samples", [])

        for i, (bounds, ratio) in enumerate(entry["bounds"]):
            if not (0.2 <= ratio <= 0.8):
                continue

            print(i, bounds, ratio)
            previous_samples = [] if reset else ([s for s in entry.get("samples", [])
                                                 if s["sample_size"] == sample_size and s["bounds"] == bounds])
            bounded_domain = Domain(density.domain.variables, density.domain.var_types, bounds)

            for j in range(n - len(previous_samples)):
                seed = seeds[j]
                samples_filename = "{}{}{}.{}.{}.{}.sample.npy".format(samples_dir, os.path.sep, name, sample_size,
                                                                       seed, i)
                labels_filename = "{}{}{}.{}.{}.{}.labels.npy".format(samples_dir, os.path.sep, name, sample_size, seed,
                                                                      i)

                if not os.path.exists(os.path.dirname(samples_filename)):
                    os.makedirs(os.path.dirname(samples_filename))

                random.seed(seed)
                np.random.seed(seed)
                samples = uniform(bounded_domain, sample_size)
                labels = evaluate(bounded_domain, density.support, samples)
                np.save(samples_filename, samples)
                np.save(labels_filename, labels)

                samples_dict[name].append({
                    "bounds": bounds,
                    "seed": seed,
                    "samples_filename": samples_filename,
                    "labels_filename": labels_filename,
                    "sample_size": sample_size
                })

        pysmt.environment.pop_env()

    def edit(summary):
        for _n, _s in samples_dict.items():
            summary[_n]["samples"] = _s

    edit_summary(edit)


def prepare_synthetic():
    pass
