import glob
import json
import os

from typing import List

import pickledb
from pywmi import RejectionEngine, nested_to_smt, import_domain
from pywmi.domain import Density, Domain

from .prepare import select_benchmark_files, benchmark_filter, get_synthetic_db
from incal.util.options import Experiment
from incal.util import analyze as show

from .learn import get_experiment

import pysmt.shortcuts as smt
import pysmt.environment


class Properties(object):
    bounds = dict()
    db = None

    @staticmethod
    def to_name(filename):
        return filename[filename.find("QF_LRA"):filename.find("smt2")+4]

    @staticmethod
    def to_sample_name(filename):
        return filename[filename.find("QF_LRA"):]

    @staticmethod
    def to_synthetic_name(filename):
        parts = os.path.basename(filename).split(".")
        return parts[0]

    @staticmethod
    def compute(experiments):
        Properties.db = pickledb.load('example.db', True)
        if Properties.db.exists("bounds"):
            Properties.bounds = Properties.db.get("bounds")
        else:
            used_names = {Properties.to_sample_name(e.parameters.original_values["data"]) for e in experiments}
            names_to_bounds = dict()
            summary_file = "remote_res/smt_lib_benchmark/qf_lra_summary.pickle"
            for name, entry, density_filename in select_benchmark_files(benchmark_filter, summary_file):
                if "samples" in entry:
                    for s in entry["samples"]:
                        name = Properties.to_sample_name(s["samples_filename"])
                        if name in used_names:
                            names_to_bounds[name] = s["bounds"]
            Properties.bounds = names_to_bounds
            Properties.db.set("bounds", Properties.bounds)

    @staticmethod
    def get_bound(experiment):
        return Properties.bounds[Properties.to_sample_name(experiment.parameters.original_values["data"])]

    @staticmethod
    def get_db_synthetic(experiment):
        return get_synthetic_db(os.path.dirname(experiment.parameters.original_values["domain"]))

    @staticmethod
    def original_h(experiment):
        db = Properties.get_db_synthetic(experiment)
        name = Properties.to_synthetic_name(experiment.imported_from_file)
        return db.get(name)["generation"]["h"]

    @staticmethod
    def accuracy_approx(experiment):
        key = "accuracy_approx:{}".format(experiment.imported_from_file)
        if Properties.db.exists(key):
            return Properties.db.get(key)
        else:
            pysmt.environment.push_env()
            pysmt.environment.get_env().enable_infix_notation = True
            if os.path.basename(experiment.imported_from_file).startswith("synthetic"):
                db = Properties.get_db_synthetic(experiment)
                name = Properties.to_synthetic_name(experiment.imported_from_file)
                entry = db.get(name)
                domain = import_domain(json.loads(entry["domain"]))
                true_formula = nested_to_smt(entry["formula"])
            else:
                density = Density.import_from(experiment.parameters.original_values["domain"])
                domain = Domain(density.domain.variables, density.domain.var_types, Properties.get_bound(experiment))
                true_formula = density.support
            learned_formula = nested_to_smt(experiment.results.formula)
            engine = RejectionEngine(domain, smt.TRUE(), smt.Real(1.0), 100000)
            accuracy = engine.compute_probability(smt.Iff(true_formula, learned_formula))
            pysmt.environment.pop_env()
            print(accuracy)
            Properties.db.set(key, accuracy)
        return accuracy


def register_derived(experiment):
    experiment.register_derived("accuracy_approx", Properties.accuracy_approx)
    experiment.register_derived("original_h", Properties.original_h)
    return experiment


def analyze(results_directories, res_path, show_args):
    experiments = []  # type: List[Experiment]
    for results_directory in results_directories:
        for filename in glob.glob("{}/**/*.result".format(results_directory), recursive=True):
            log_file = filename.replace(".result", ".log")
            if not os.path.exists(log_file):
                log_file = None
            experiment = get_experiment(res_path).load(filename)
            experiments.append(register_derived(experiment))

    Properties.compute(experiments)
    show.show(experiments, *show_args)
