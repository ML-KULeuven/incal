from __future__ import print_function, division

import argparse
import random

import os

import numpy as np
import pysmt.shortcuts as smt
import time

from pywmi.export import Exportable
from typing import List

from incal import Formula
from incal.learner import Learner
from pywmi import Domain, evaluate
from pywmi.sample import uniform


class InsufficientBalanceError(RuntimeError):
    def __init__(self, partial_samples=None):
        self.partial_samples = partial_samples


class SyntheticFormula(Formula):
    def __init__(self, domain, support, cnf_or_dnf, formula_count, terms_per_formula, half_space_count, name=None):
        super().__init__(domain, support)
        self.cnf_or_dnf = cnf_or_dnf
        self.formula_count = formula_count
        self.terms_per_formula = terms_per_formula
        self.half_space_count = half_space_count
        self.name = name

    @property
    def bool_count(self):
        return len(self.domain.bool_vars)

    @property
    def real_count(self):
        return len(self.domain.real_vars)

    @property
    def k(self):
        return self.formula_count

    @property
    def h(self):
        return self.half_space_count

    @property
    def literals(self):
        return self.terms_per_formula

    def get_data(self, sample_count, max_ratio):
        samples, labels = get_problem_samples(self.domain, self.support, sample_count, max_ratio)
        return SyntheticData(self, samples, labels)

    def get_state(self):
        state = Formula.get_state(self)
        state["cnf_or_dnf"] = self.cnf_or_dnf
        state["formula_count"] = self.formula_count
        state["terms_per_formula"] = self.terms_per_formula
        state["half_space_count"] = self.half_space_count
        state["name"] = self.name
        return state

    @classmethod
    def from_state(cls, state: dict):
        formula = Formula.from_state(state)
        return SyntheticFormula(
            formula.domain,
            formula.support,
            state["cnf_or_dnf"],
            state["formula_count"],
            state["terms_per_formula"],
            state["half_space_count"],
            state["name"],
        )


class SyntheticData(Exportable):
    def __init__(self, formula, samples, labels):
        # type: (SyntheticFormula, np.ndarray, np.ndarray) -> None
        self.formula = formula
        self.samples = samples
        self.labels = labels

    def get_state(self):
        return {
            "formula": self.formula.get_state(),
            "samples": self.samples.tolist(),
            "labels": self.labels.tolist()
        }

    @classmethod
    def from_state(cls, state: dict):
        return cls(
            SyntheticFormula.from_state(state["formula"]),
            np.array(state["samples"]),
            np.array(state["labels"]),
        )


def generate_half_space_sample(domain, real_count):
    samples = uniform(domain, real_count)
    coefficients, offset = Learner.fit_hyperplane(domain, samples)
    coefficients = [smt.Real(float(coefficients[i][0])) * domain.get_symbol(domain.real_vars[i]) for i in
                    range(real_count)]
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

    return Domain(variables, var_types, var_domains)


def get_problem_samples(domain, support, sample_count, max_ratio):
    minimal_count = sample_count * min(max_ratio, 1 - max_ratio)
    samples = uniform(domain, sample_count)
    labels = evaluate(domain, support, samples)
    positive_count = sum(labels)
    if positive_count < minimal_count or (sample_count - positive_count) < minimal_count:
        raise InsufficientBalanceError()

    return samples, labels


def get_synthetic_problem_name(prefix, bool_count, real_count, cnf_or_dnf, k, l_per_term, h, sample_count, seed,
                               ratio_percent, i=None):
    name = "{prefix}_{bc}_{rc}_{type}_{fc}_{tpf}_{hc}_{sc}_{seed}_{ratio}" \
        .format(bc=bool_count, rc=real_count, type=cnf_or_dnf, fc=k, tpf=l_per_term, hc=h, sc=sample_count,
                prefix=prefix, seed=seed, ratio=int(ratio_percent))
    if i is not None:
        name = name + "_" + str(i)
    return name


class GeneratorError(RuntimeError):
    pass


class Generator(object):
    def __init__(self, bool_count, real_count, bias, k, l, h, sample_count, max_ratio, seed, prefix):
        self.domain = generate_domain(bool_count, real_count)
        self.bias = bias
        self.k = k
        self.l = l
        self.h = h
        self.sample_count = sample_count
        self.max_ratio = max_ratio
        self.seed = seed
        self.prefix = prefix
        self.max_tries = 1000

    @property
    def bool_count(self):
        return len(self.domain.bool_vars)

    @property
    def real_count(self):
        return len(self.domain.real_vars)

    def symbol(self, var_name):
        return self.domain.get_symbol(var_name)

    def test_ratio(self, labels):
        max_ratio = max(self.max_ratio, 1 - self.max_ratio)
        return (1 - max_ratio) <= sum(labels) / self.sample_count <= max_ratio

    def get_name(self):
        b = self.bool_count
        r = self.real_count
        ratio = self.max_ratio * 100
        k, l, h = self.k, self.l, self.h
        return get_synthetic_problem_name(self.prefix, b, r, self.bias, k, l, h, self.sample_count, self.seed, ratio)

    def get_samples(self):
        return uniform(self.domain, self.sample_count)

    def get_half_spaces(self, samples):
        half_spaces = []
        print("Generating half spaces: ", end="")
        if self.real_count > 0:
            while len(half_spaces) < self.h:
                half_space = generate_half_space_sample(self.domain, self.real_count)
                labels = evaluate(self.domain, half_space, samples)
                half_spaces.append((half_space, labels))
                print("y", end="")

        print()
        return half_spaces

    def get_term(self, literal_pool):
        print("Generate term: ", end="")
        for i in range(self.max_tries):
            literals = random.sample(literal_pool, self.l)

            term = smt.Or(*list(zip(*literals))[0])

            covered = np.zeros(self.sample_count)
            significant_literals = 0
            for _, labels in literals:
                prev_size = sum(covered)
                covered = np.logical_or(covered, labels)
                if (sum(covered) - prev_size) / self.sample_count >= 0.05:
                    significant_literals += 1

            if significant_literals == self.l:  # & test_ratio(covered):
                print("y", end="")
                print()
                return term, covered
            else:
                print("x", end="")
        print(" Failed after {} tries".format(self.max_tries))
        raise GeneratorError()

    def get_formula(self, name, literal_pool):
        print("Generate formula:")
        for i in range(self.max_tries):
            terms = [self.get_term(literal_pool) for _ in range(self.k)]
            formula = smt.And(*list(zip(*terms))[0])

            covered = np.ones(self.sample_count)
            significant_terms = 0
            for _, labels in terms:
                prev_size = sum(covered)
                covered = np.logical_and(covered, labels)
                if (prev_size - sum(covered)) / self.sample_count >= 0.05:
                    significant_terms += 1

            if significant_terms == self.k and self.test_ratio(covered):
                print("y({:.2f})".format(sum(covered) / self.sample_count), end="")
                return SyntheticFormula(self.domain, formula, "cnf", self.k, self.l, self.h, name)
            else:
                if significant_terms == self.k:
                    print("Ratio not satisfied")
                else:
                    print("Not enough significant terms")
        print("Failed to generate formula after {} tries".format(self.max_tries))
        raise GeneratorError()

    def generate_formula(self):
        for i in range(self.max_tries):
            print("Generating formula...")
            samples = self.get_samples()
            half_spaces = self.get_half_spaces(self.get_samples())
            bool_literals = [(self.symbol(v), samples[:, self.domain.variables.index(v)])
                             for v in self.domain.bool_vars]
            literal_pool = half_spaces + bool_literals
            literal_pool += [(smt.Not(l), np.logical_not(bits)) for l, bits in literal_pool]
            print([(v, len(l)) for v, l in literal_pool])

            try:
                formula = self.get_formula(self.get_name(), literal_pool)
                return formula
            except GeneratorError:
                continue

        raise RuntimeError("Could not generate a formula within {} tries".format(self.max_tries))

    def generate_data_set(self):
        for i in range(self.max_tries):
            try:
                return self.generate_formula().get_data(self.sample_count, self.max_ratio)
            except InsufficientBalanceError:
                pass
        raise RuntimeError("Could not generate a dataset within {} tries".format(self.max_tries))


def get_seed():
    return random.randint(0, 2 ** 32 - 1)


def generate_synthetic_formula(
        prefix: str,
        boolean_count: int,
        real_count: int,
        cnf_or_dnf: str,
        formula_count: int,
        terms_per_formula: int,
        half_space_count: int,
        sample_count: int,
        ratio_percent: int,
        seed=None,
) -> SyntheticFormula:
    seed = seed or get_seed()
    random.seed(seed)
    np.random.seed(seed)

    ratio = ratio_percent / 100
    producer = Generator(boolean_count, real_count, cnf_or_dnf, formula_count, terms_per_formula, half_space_count,
                         sample_count, ratio, seed, prefix)
    return producer.generate_formula()


def generate_synthetic_dataset(
        prefix: str,
        boolean_count: int,
        real_count: int,
        cnf_or_dnf: str,
        formula_count: int,
        terms_per_formula: int,
        half_space_count: int,
        sample_count: int,
        ratio_percent: int,
        seed=None,
) -> SyntheticData:
    seed = seed or get_seed()
    random.seed(seed)
    np.random.seed(seed)

    ratio = ratio_percent / 100
    producer = Generator(boolean_count, real_count, cnf_or_dnf, formula_count, terms_per_formula, half_space_count,
                         sample_count, ratio, seed, prefix)
    return producer.generate_data_set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--problems", default=None, type=int)
    parser.add_argument("--datasets", default=None, type=int)
    parser.add_argument("--prefix", default="synthetic")
    parser.add_argument("--bool_count", default=2, type=int)
    parser.add_argument("--real_count", default=2, type=int)
    parser.add_argument("--bias", default="cnf")
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--literals", default=4, type=int)
    parser.add_argument("--h", default=7, type=int)
    parser.add_argument("--sample_count", default=1000, type=int)
    parser.add_argument("--ratio", default=90, type=int)
    parsed = parser.parse_args()

    data_dir = parsed.data_dir
    print(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if parsed.problems:
        for _ in range(parsed.problems):
            formula = generate_synthetic_formula(
                parsed.prefix, parsed.bool_count, parsed.real_count, parsed.bias, parsed.k,
                parsed.literals, parsed.h, parsed.sample_count, parsed.ratio
            )
            formula.to_file(os.path.join(data_dir, formula.name + ".json"))

    if parsed.datasets:
        for _ in range(parsed.datasets):
            dataset = generate_synthetic_dataset(
                parsed.prefix, parsed.bool_count, parsed.real_count, parsed.bias, parsed.k,
                parsed.literals, parsed.h, parsed.sample_count, parsed.ratio
            )
            dataset.to_file(os.path.join(data_dir, dataset.formula.name + ".data.json"))


