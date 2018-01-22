from __future__ import print_function, division

import argparse
import fnmatch
import hashlib
import json

import os
import random

import pysmt.shortcuts as smt
import re

import time

import generator
import inc_logging
import parse
import problem
from incremental_learner import RandomViolationsStrategy
from k_cnf_smt_learner import KCnfSmtLearner
from parameter_free_learner import learn_bottom_up
from smt_walk import SmtWalker


class OperatorWalker(SmtWalker):
    def __init__(self):
        self.operators = set()

    def walk_and(self, args):
        self.operators.add("&")
        self.walk_smt_multiple(args)

    def walk_or(self, args):
        self.operators.add("|")
        self.walk_smt_multiple(args)

    def walk_plus(self, args):
        self.operators.add("+")
        self.walk_smt_multiple(args)

    def walk_minus(self, left, right):
        self.operators.add("-")
        self.walk_smt_multiple([left, right])

    def walk_times(self, args):
        self.operators.add("*")
        self.walk_smt_multiple(args)

    def walk_not(self, argument):
        self.operators.add("~")
        self.walk_smt_multiple([argument])

    def walk_ite(self, if_arg, then_arg, else_arg):
        self.operators.add("ite")
        self.walk_smt_multiple([if_arg, then_arg, else_arg])

    def walk_pow(self, base, exponent):
        self.operators.add("^")
        self.walk_smt_multiple([base, exponent])

    def walk_lte(self, left, right):
        self.operators.add("<=")
        self.walk_smt_multiple([left, right])

    def walk_lt(self, left, right):
        self.operators.add("<")
        self.walk_smt_multiple([left, right])

    def walk_equals(self, left, right):
        self.operators.add("=")
        self.walk_smt_multiple([left, right])

    def walk_symbol(self, name, v_type):
        pass

    def walk_constant(self, value, v_type):
        pass

    def find_operators(self, formula):
        self.walk_smt(formula)
        return list(self.operators)


class HalfSpaceWalker(SmtWalker):
    def __init__(self):
        self.half_spaces = set()

    def walk_and(self, args):
        self.walk_smt_multiple(args)

    def walk_or(self, args):
        self.walk_smt_multiple(args)

    def walk_plus(self, args):
        self.walk_smt_multiple(args)

    def walk_minus(self, left, right):
        self.walk_smt_multiple([left, right])

    def walk_times(self, args):
        self.walk_smt_multiple(args)

    def walk_not(self, argument):
        self.walk_smt_multiple([argument])

    def walk_ite(self, if_arg, then_arg, else_arg):
        self.walk_smt_multiple([if_arg, then_arg, else_arg])

    def walk_pow(self, base, exponent):
        self.walk_smt_multiple([base, exponent])

    def walk_lte(self, left, right):
        self.half_spaces.add(parse.smt_to_nested(left <= right))

    def walk_lt(self, left, right):
        self.half_spaces.add(parse.smt_to_nested(left < right))

    def walk_equals(self, left, right):
        self.walk_smt_multiple([left, right])

    def walk_symbol(self, name, v_type):
        pass

    def walk_constant(self, value, v_type):
        pass

    def find_half_spaces(self, formula):
        self.walk_smt(formula)
        return list(self.half_spaces)


def import_problem(name, filename):
    target_formula = smt.read_smtlib(filename)

    variables = target_formula.get_free_variables()
    var_names = [str(v) for v in variables]
    var_types = {str(v): v.symbol_type() for v in variables}
    var_domains = {str(v): (0, 200) for v in variables}  # TODO This is a hack

    domain = problem.Domain(var_names, var_types, var_domains)
    target_problem = problem.Problem(domain, target_formula, name)

    return target_problem


def get_cache_name():
    cache = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo"), "cache")
    return os.path.join(cache, "summary.json")


def get_problem_file(problem_id):
    return os.path.join(os.path.dirname(get_cache_name()), "problems", "{}.txt".format(problem_id))


def load():
    name = get_cache_name()
    if not os.path.exists(name):
        with open(name, "w") as f:
            json.dump({"files": {}}, f)

    with open(name, "r") as f:
        flat = json.load(f)
    return flat


def dump(flat):
    name = get_cache_name()
    temp_name = "{}.tmp".format(name)
    with open(temp_name, "w") as f:
        json.dump(flat, f)
    os.rename(temp_name, name)


def scan(full_dir, root_dir):
    matches = []

    for root, dir_names, file_names in os.walk(full_dir):
        for filename in fnmatch.filter(file_names, '*.smt2'):
            full_path = os.path.abspath(os.path.join(root, filename))
            matches.append(re.match("{r}{s}(.*)".format(s=os.path.sep, r=root_dir), full_path).group(1))

    for match in matches:
        print(match)

    flat = load()
    problems = flat["files"]
    counter = 0
    for match in matches:
        if match not in problems:
            print("Importing {}".format(match))
            smt_file = os.path.join(root_dir, match)
            if os.path.getsize(smt_file) / 1024 <= 100:
                imported_problem = import_problem(match, smt_file)
                problems[match] = {
                    "loaded": True,
                    "var_count": len(imported_problem.domain.variables),
                    "file_size": os.path.getsize(match)
                }
                smt.reset_env()
            else:
                problems[match] = {"loaded": False, "reason": "size", "file_size": os.path.getsize(match)}
            counter += 1
            if counter >= 1:
                dump(flat)
                counter = 0

    if counter != 0:
        dump(flat)


def has_equals(_props):
    return "=" in [str(o) for o in _props["operators"]]


def has_conjunctions(_props):
    return "&" in [str(o) for o in _props["operators"]]


def has_disjunctions(_props):
    return "|" in [str(o) for o in _props["operators"]]


def has_connectives(_props):
    return has_conjunctions(_props) or has_disjunctions(_props)


def var_count(_props):
    return _props["var_count"]


def half_space_count(_props):
    return len(_props["half_spaces"])


def file_size(_props):
    return _props["file_size"]


def analyze(root_dir):
    flat = load()
    files = flat["files"]
    if "lookup" not in flat:
        flat["lookup"] = dict()
    lookup = flat["lookup"]

    print(*["mode", "var_count", "half_space_count", "file_size", "name"], sep="\t")
    for name, properties in files.items():
        smt_file = os.path.join(root_dir, name)
        if properties["loaded"] and properties["var_count"] < 10:
            change = False
            if "id" not in properties:
                properties["id"] = hashlib.sha256(smt_file).hexdigest()
                change = True

            if properties["id"] not in lookup:
                lookup[properties["id"]] = name
                change = True

            problem_file = get_problem_file(properties["id"])
            if not os.path.isfile(problem_file):
                with open(problem_file, "w") as f:
                    print(problem.export_problem(import_problem(name, smt_file)), file=f)

            with open(problem_file, "r") as f:
                target_problem = problem.import_problem(json.load(f))

            if "operators" not in properties:
                properties["operators"] = OperatorWalker().find_operators(target_problem.theory)
                change = True

            if "half_spaces" not in properties:
                properties["half_spaces"] = HalfSpaceWalker().find_half_spaces(target_problem.theory)
                change = True

            if change:
                dump(flat)

            if not has_equals(properties) and has_connectives(properties):
                mode = "D" if has_disjunctions(properties) else "C"
                information = [mode, var_count(properties), half_space_count(properties), file_size(properties), name]
                print(*[str(i) for i in information], sep="\t")


def learn_formula(problem_id, domain, h, data, seed):
    initial_size = 20
    violations_size = 10
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "results")

    def learn_inc(_data, _k, _h):
        initial_indices = random.sample(list(range(len(data))), initial_size)
        violations_strategy = RandomViolationsStrategy(violations_size)
        learner = KCnfSmtLearner(_k, _h, violations_strategy)
        log_file = os.path.join(log_dir, "{}_{}_{}_{}_{}.txt".format(problem_id, len(data), seed, _k, _h))
        learner.add_observer(inc_logging.LoggingObserver(log_file, seed, True, violations_strategy))
        learned_theory = learner.learn(domain, data, initial_indices)
        # learned_theory = Or(*[And(*planes) for planes in hyperplane_dnf])
        print("Learned theory:\n{}".format(parse.smt_to_nested(learned_theory)))
        return learned_theory

    phi, k, h = learn_bottom_up(data, learn_inc, 1, 1, init_h=h, max_h=h)

    overview = os.path.join(log_dir, "problems.txt")
    if not os.path.isfile(overview):
        flat = {}
    else:
        with open(overview, "r") as f:
            flat = json.load(f)
    if problem_id not in flat:
        flat[problem_id] = {}
    flat[problem_id][len(data)] = {"k": k, "h": h, "seed": seed}
    with open(overview, "w") as f:
        json.dump(flat, f)


def learn(sample_count):
    flat = load()
    files = flat["files"]
    ratio_dict = flat["ratios"]
    seed = time.time()
    for name, props in files.items():
        if props["loaded"] and props["var_count"] < 10 and not has_equals(props) and has_disjunctions(props) and \
                ratio_dict[name]["finite"] and 0.1 <= ratio_dict[name]["ratio"] <= 0.9:
            with open(get_problem_file(props["id"]), "r") as f:
                target_problem = problem.import_problem(json.load(f))
            adapted_problem = adapt_domain(target_problem, ratio_dict[name]["lb"], ratio_dict[name]["ub"])
            samples = generator.get_problem_samples(adapted_problem, sample_count, 1)
            learn_formula(props["id"], adapted_problem.domain, len(props["half_spaces"]), samples, seed)
            print(props["id"], name)


def adapt_domain(target_problem, lb, ub):
    domain = target_problem.domain
    var_domains = {}
    for v in domain.variables:
        var_domains[v] = (lb, ub)
    adapted_domain = problem.Domain(domain.variables, domain.var_types, var_domains)
    return problem.Problem(adapted_domain, target_problem.theory, target_problem.name)


def ratios():
    flat = load()
    files = flat["files"]
    if "ratios" not in flat:
        flat["ratios"] = dict()
    ratio_dict = flat["ratios"]

    sample_count = 1000
    for name, props in files.items():
        if props["loaded"] and props["var_count"] < 10 and not has_equals(props) and has_disjunctions(props) and \
                name not in ratio_dict:
            with open(get_problem_file(props["id"]), "r") as f:
                target_problem = problem.import_problem(json.load(f))

            lb, ub = -1000, 1000
            current_problem = adapt_domain(target_problem, lb, ub)
            samples = generator.get_problem_samples(current_problem, sample_count, 1)
            positive_count = len([x for x, y in samples if y])
            is_finite = positive_count not in [0, sample_count]
            ratio_dict[name] = {"finite": is_finite, "samples": sample_count, "lb": lb, "ub": ub}
            if is_finite:
                ratio_dict[name]["ratio"] = positive_count / sample_count
            print(name, ratio_dict[name])

    dump(flat)


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("filename")
        parser.add_argument("learning_samples", type=int)
        args = parser.parse_args()

        full_dir = os.path.abspath(args.filename)
        root_dir = os.path.dirname(full_dir)

        # scan(full_dir, root_dir)
        # analyze(root_dir)
        # ratios()
        learn(args.learning_samples)

    parse_args()
