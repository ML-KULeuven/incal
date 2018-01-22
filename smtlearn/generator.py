from __future__ import print_function

import json
import random

import os
import pysmt.shortcuts as smt
import time

import problem
import parse
from learner import Learner
from smt_check import SmtChecker


class InsufficientBalanceError(RuntimeError):
    def __init__(self, partial_samples=None):
        self.partial_samples = partial_samples


class SyntheticProblem(object):
    def __init__(self, theory_problem, cnf_or_dnf, formula_count, terms_per_formula, half_space_count):
        self.theory_problem = theory_problem
        self.cnf_or_dnf = cnf_or_dnf
        self.formula_count = formula_count
        self.terms_per_formula = terms_per_formula
        self.half_space_count = half_space_count

    @property
    def bool_count(self):
        return len(self.theory_problem.domain.bool_vars)

    @property
    def real_count(self):
        return len(self.theory_problem.domain.real_vars)

    def get_data(self, sample_count, max_ratio):
        samples = get_problem_samples(self.theory_problem, sample_count, max_ratio)
        return SyntheticData(self, samples)

    @staticmethod
    def create(domain, cnf_or_dnf, formula_count, terms_per_formula, half_space_count, name):
        if cnf_or_dnf == "cnf":
            theory = generate_cnf(domain, formula_count, terms_per_formula, half_space_count)
        elif cnf_or_dnf == "dnf":
            theory = generate_dnf(domain, formula_count, terms_per_formula, half_space_count)
        elif cnf_or_dnf == "cnf_strict":
            theory = generate_strict_cnf(domain, formula_count, terms_per_formula, half_space_count)
        else:
            raise RuntimeError("cnf_or_dnf was neither 'cnf' nor 'dnf'")
        theory_problem = problem.Problem(domain, theory, name)
        return SyntheticProblem(theory_problem, cnf_or_dnf, formula_count, terms_per_formula, half_space_count)


class Data(object):
    def __init__(self, theory_problem, samples):
        self.theory_problem = theory_problem
        self.samples = samples


class SyntheticData(object):
    def __init__(self, synthetic_problem, samples):
        self.synthetic_problem = synthetic_problem
        self.samples = samples


def export_synthetic_problem(synthetic_problem, to_str=True):
    """
    :type synthetic_problem: SyntheticProblem
    :type to_str: bool
    """
    flat = {
        "problem": problem.export_problem(synthetic_problem.theory_problem, to_str=False),
        "cnf_or_dnf": synthetic_problem.cnf_or_dnf,
        "formula_count": synthetic_problem.formula_count,
        "terms_per_formula": synthetic_problem.terms_per_formula,
        "half_space_count": synthetic_problem.half_space_count,
    }
    return json.dumps(flat) if to_str else flat


def import_synthetic_problem(flat):
    theory_problem = problem.import_problem(flat["problem"])
    cnf_or_dnf = flat["cnf_or_dnf"]
    formula_count = flat["formula_count"]
    terms_per_formula = flat["terms_per_formula"]
    half_space_count = flat["half_space_count"]
    return SyntheticProblem(theory_problem, cnf_or_dnf, formula_count, terms_per_formula, half_space_count)


def export_data(data, to_str=True):
    """
    :type data: Data
    :type to_str: bool
    """
    flat_samples = []
    for row, label in data.samples:
        flat_samples.append({"instance": row, "label": label})

    flat = {
        "problem": problem.export_problem(data.theory_problem, to_str=False),
        "samples": flat_samples,
    }
    return json.dumps(flat) if to_str else flat


def import_data(flat):
    theory_problem = problem.import_problem(flat["problem"])
    samples = []
    for example in flat["samples"]:
        samples.append((example["instance"], example["label"]))
    return Data(theory_problem, samples)


def export_synthetic_data(synthetic_data, to_str=True):
    """
    :type synthetic_data: SyntheticData
    :type to_str: bool
    """
    flat_samples = []
    synthetic_problem = synthetic_data.synthetic_problem
    for row, label in synthetic_data.samples:
        flat_samples.append({"instance": row, "label": label})

    flat = {
        "synthetic_problem": export_synthetic_problem(synthetic_problem, to_str=False),
        "samples": flat_samples,
    }
    return json.dumps(flat) if to_str else flat


def import_synthetic_data(flat):
    synthetic_problem = import_synthetic_problem(flat["synthetic_problem"])
    samples = []
    for example in flat["samples"]:
        samples.append((example["instance"], example["label"]))
    return SyntheticData(synthetic_problem, samples)


def generate_cnf(domain, and_count, or_count, half_space_count):
    formulas = get_formulas(domain, and_count, or_count, half_space_count)
    return smt.And(smt.Or(*formula) for formula in formulas)


def generate_strict_cnf(domain, and_count, or_count, half_space_count):
    half_spaces = [generate_half_space_sample(domain, len(domain.real_vars)) for _ in range(half_space_count)]
    candidates = [domain.get_symbol(v) for v in domain.bool_vars] + half_spaces
    candidates += [smt.Not(c) for c in candidates]

    formulas = []
    iteration = 0
    max_iterations = 100 * and_count
    while len(formulas) < and_count:
        if iteration >= max_iterations:
            return generate_strict_cnf(domain, and_count, or_count, half_space_count)
        iteration += 1
        formula_candidates = [c for c in candidates]
        random.shuffle(formula_candidates)
        formula = []
        try:
            while len(formula) < or_count:
                next_term = formula_candidates.pop(0)
                if len(formula) == 0 or smt.is_sat(~smt.Or(*formula) & next_term):
                    formula.append(next_term)
        except IndexError:
            continue
        if len(formulas) == 0 or smt.is_sat(~smt.And(*[smt.Or(*f) for f in formulas]) & smt.Or(*formula)):
            formulas.append(formula)
    return smt.And(*[smt.Or(*f) for f in formulas])


def generate_dnf(domain, or_count, and_count, half_space_count):
    formulas = get_formulas(domain, and_count, or_count, half_space_count)
    return smt.Or(smt.And(*formula) for formula in formulas)


def get_formulas(domain, formula_count, terms_per_formula, half_space_count):
    half_spaces = [generate_half_space_sample(domain, len(domain.real_vars)) for _ in range(half_space_count)]
    candidates = [domain.get_symbol(v) for v in domain.bool_vars] + half_spaces
    return [
        list(random.sample(candidates, terms_per_formula))
        for _ in range(formula_count)
    ]


def generate_half_space(domain, real_count):
    coefficients = [smt.Real(random.random() * 2 - 1) * domain.get_symbol(domain.real_vars[i]) for i in range(real_count)]
    return smt.LE(smt.Plus(*coefficients), smt.Real(random.random() * 2 - 1))


def generate_half_space_sample(domain, real_count):
    samples = [get_sample(domain) for _ in range(real_count)]
    coefficients, offset = Learner.fit_hyperplane(domain, samples)
    coefficients = [smt.Real(float(coefficients[i][0])) * domain.get_symbol(domain.real_vars[i]) for i in range(real_count)]
    if random.random() < 0.5:
        return smt.Plus(*coefficients) <= offset
    else:
        return smt.Plus(*coefficients) >= offset


def generate_domain(bool_count, real_count):
    variables = ["b{}".format(i) for i in range(bool_count)] + ["r{}".format(i) for i in range(real_count)]
    var_types = dict()
    var_domains = dict()
    for i, v in enumerate(variables):
        if i < bool_count:
            var_types[v] = smt.BOOL
        else:
            var_types[v] = smt.REAL
            var_domains[v] = (0, 1)

    return problem.Domain(variables, var_types, var_domains)


def evaluate_assignment(test_problem, assignment):
    """
    Calculate if the given assignment satisfies the problem formula
    :param test_problem: The problem containing the formula and domain
    :param assignment: An assignment to (all) the domain variables
    :return: True iff the assignment satisfies the problem formula
    """
    return SmtChecker(assignment).walk_smt(test_problem.theory)


def get_labeled_sample(test_problem):
    """
    Sample n instances for the given problem
    :param test_problem: The problem to sample from
    :return: A sample (dictionary mapping variable names to values) and its label (True iff the sample satisfies the
    theory and False otherwise)
    """
    instance = get_sample(test_problem.domain)
    return instance, evaluate_assignment(test_problem, instance)


def get_sample(domain):
    instance = dict()
    for v in domain.variables:
        if domain.var_types[v] == smt.REAL:
            lb, ub = domain.var_domains[v]
            instance[v] = random.uniform(lb, ub)
        elif domain.var_types[v] == smt.BOOL:
            instance[v] = True if random.random() < 0.5 else False
        else:
            raise RuntimeError("Unknown variable type {} for variable {}", domain.var_types[v], v)
    return instance


def get_problem_samples(test_problem, sample_count, max_ratio):
    samples = []
    pos_count = 0
    neg_count = 0
    minimal_count = sample_count * min(max_ratio, 1 - max_ratio)
    for i in range(sample_count):
        instance, label = get_labeled_sample(test_problem)
        samples.append((instance, label))
        if label:
            pos_count += 1
        else:
            neg_count += 1
        remaining = sample_count - (pos_count + neg_count)

        # Approximate check
        # if pos_count + neg_count == 20:
        #     current_ratio = max(pos_count, neg_count) / float(pos_count + neg_count)
        #     if current_ratio > max_ratio + 0.1:
        #         print("Triggered")
        #         raise InsufficientBalanceError()

        # Definite check
        if pos_count + remaining < minimal_count or neg_count + remaining < minimal_count:
            print("Definite test")
            raise InsufficientBalanceError()
    return samples


def generate_synthetic_data(data_sets_per_setting, bool_count, real_count, cnf_or_dnf, formula_count, terms_per_formula,
                            half_space_count, sample_count, max_ratio, prefix="synthetic"):
    data_set_count = 0
    domain = generate_domain(bool_count, real_count)
    while data_set_count < data_sets_per_setting:
        try:
            name = "{prefix}_{bc}_{rc}_{type}_{fc}_{tpf}_{hc}_{sc}_{i}"\
                .format(bc=bool_count, rc=real_count, type=cnf_or_dnf, fc=formula_count, tpf=terms_per_formula,
                        hc=half_space_count, sc=sample_count, i=data_set_count, prefix=prefix)
            data = SyntheticProblem.create(domain, cnf_or_dnf, formula_count, terms_per_formula, half_space_count,
                                           name).get_data(sample_count, max_ratio)
            data_set_count += 1
            yield data
        except InsufficientBalanceError:
            pass


def import_synthetic_data_files(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r") as f:
                yield import_synthetic_data(json.load(f))


def test():
    random.seed(666)
    prefix = "synthetic"
    for data_set in generate_synthetic_data(10, 0, 2, "cnf_strict", 3, 4, 7, 1000, 0.8, prefix):
        print(data_set.synthetic_problem.theory_problem.name)
        file_name = "../data/{}.txt".format(data_set.synthetic_problem.theory_problem.name)
        with open(file_name, "w") as f:
            print(export_synthetic_data(data_set), file=f)

    # imported = list(import_synthetic_data_files("../data", prefix))
    # print(imported[0].synthetic_problem.theory_problem.name)


if __name__ == "__main__":
    test()
