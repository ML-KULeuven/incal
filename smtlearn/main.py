from __future__ import print_function

import itertools
import random
import time
from math import sqrt

import matplotlib as mpl
from pysmt.shortcuts import *

import plotting
from incremental_learner import AllViolationsStrategy, RandomViolationsStrategy
from k_cnf_smt_learner import KCnfSmtLearner
from k_dnf_logic_learner import KDNFLogicLearner, GreedyLogicDNFLearner, GreedyMaxRuleLearner
from k_dnf_smt_learner import KDnfSmtLearner
from k_dnf_greedy_learner import GreedyMilpRuleLearner
from parameter_free_learner import learn_bottom_up
from problem import Domain, Problem
from smt_check import SmtChecker

import parse

mpl.use('TkAgg')
import matplotlib.pyplot as plt

from k_dnf_learner import KDNFLearner


def xy_domain():
    variables = ["x", "y"]
    var_types = {"x": REAL, "y": REAL}
    var_domains = {"x": (0, 1), "y": (0, 1)}
    return Domain(variables, var_types, var_domains)


def simple_checker_problem():

    theory = Or(
        And(LE(Symbol("x", REAL), Real(0.5)), LE(Symbol("y", REAL), Real(0.5))),
        And(GT(Symbol("x", REAL), Real(0.5)), GT(Symbol("y", REAL), Real(0.5)))
    )

    return Problem(xy_domain(), theory, "simple_checker")


def simple_checker_problem_cnf():
    x, y = (Symbol(n, REAL) for n in ["x", "y"])
    theory = ((x <= 0.5) | (y > 0.5)) & ((x > 0.5) | (y <= 0.5))
    return Problem(xy_domain(), theory, "simple_cnf_checker")


def checker_problem():
    variables = ["x", "y", "a"]
    var_types = {"x": REAL, "y": REAL, "a": BOOL}
    var_domains = {"x": (0, 1), "y": (0, 1)}

    theory = Or(
        And(LE(Symbol("x", REAL), Real(0.5)), LE(Symbol("y", REAL), Real(0.5)), Symbol("a", BOOL)),
        And(GT(Symbol("x", REAL), Real(0.5)), GT(Symbol("y", REAL), Real(0.5)), Symbol("a", BOOL)),
        And(GT(Symbol("x", REAL), Real(0.5)), LE(Symbol("y", REAL), Real(0.5)), Not(Symbol("a", BOOL))),
        And(LE(Symbol("x", REAL), Real(0.5)), GT(Symbol("y", REAL), Real(0.5)), Not(Symbol("a", BOOL)))
    )

    return Problem(Domain(variables, var_types, var_domains), theory, "checker")


def simple_univariate_problem():
    variables = ["x"]
    var_types = {"x": REAL}
    var_domains = {"x": (0, 1)}

    theory = LE(Symbol("x", REAL), Real(0.6))

    return Problem(Domain(variables, var_types, var_domains), theory, "one_test")


def shared_hyperplane_problem():
    domain = xy_domain()
    x, y = (domain.get_symbol(v) for v in ["x", "y"])
    # y <= -x + 1.25
    shared1 = LE(y, Plus(Times(Real(-1.0), x), Real(1.25)))
    # y >= -x + 0.75
    shared2 = GE(y, Plus(Times(Real(-1.0), x), Real(0.75)))

    # y <= x + 0.5
    h1 = LE(y, Plus(x, Real(0.5)))
    # y >= x + 0.25
    h2 = GE(y, Plus(x, Real(0.25)))

    # y <= x - 0.25
    h3 = LE(y, Plus(x, Real(-0.25)))
    # y >= x - 0.5
    h4 = GE(y, Plus(x, Real(-0.5)))
    return Problem(domain, Or(And(shared1, shared2, h1, h2), And(shared1, shared2, h3, h4)), "shared")


def cross_problem():
    domain = xy_domain()
    x, y = (domain.get_symbol(v) for v in ["x", "y"])
    top = y <= 0.9
    middle_top = y <= 0.7
    middle_bottom = y >= 0.5
    bottom = y >= 0.1

    left = x >= 0.2
    middle_left = x >= 0.4
    middle_right = x <= 0.6
    right = x <= 0.8
    theory = (top & middle_left & middle_right & bottom) | (left & middle_top & middle_bottom & right)
    return Problem(domain, theory, "cross")


def bool_xor_problem():
    variables = ["a", "b"]
    var_types = {"a": BOOL, "b": BOOL}
    var_domains = dict()
    domain = Domain(variables, var_types, var_domains)

    a, b = (domain.get_symbol(v) for v in variables)

    theory = (a & ~b) | (~a & b)
    return Problem(domain, theory, "2xor")


def evaluate_assignment(problem, assignment):
    # substitution = {problem.domain.get_symbol(v): val for v, val in assignment.items()}
    # return problem.theory.substitute(substitution).simplify().is_true()
    return SmtChecker(assignment).check(problem.theory)


def sample(problem, n, seed=None):
    """
    Sample n instances for the given problem
    :param problem: The problem to sample from
    :param n: The number of samples
    :param seed: An optional seed
    :return: A list containing n samples and their label (True iff the sample satisfies the theory and False otherwise)
    """
    # TODO Other distributions
    samples = []
    for i in range(n):
        instance = dict()
        for v in problem.domain.variables:
            if problem.domain.var_types[v] == REAL:
                lb, ub = problem.domain.var_domains[v]
                instance[v] = random.uniform(lb, ub)
            elif problem.domain.var_types[v] == BOOL:
                instance[v] = True if random.random() < 0.5 else False
            else:
                raise RuntimeError("Unknown variable type {} for variable {}", problem.domain.var_types[v], v)
        samples.append((instance, evaluate_assignment(problem, instance)))
    return samples


def learn(domain, data):
    return None


def draw_points(feat_x, feat_y, data, name, hyperplanes=None, truth=None):
    if truth is None:
        truth = data

    true_pos = []
    false_pos = []
    true_neg = []
    false_neg = []

    for i in range(len(data)):
        row = data[i]
        point = (float(row[0][feat_x].constant_value()), float(row[0][feat_y].constant_value()))
        if truth[i][1] and row[1]:
            true_pos.append(point)
        elif truth[i][1] and not row[1]:
            false_neg.append(point)
        elif not truth[i][1] and not row[1]:
            true_neg.append(point)
        else:
            false_pos.append(point)

    if len(true_pos) > 0: plt.scatter(*zip(*true_pos), c="green", marker="o")
    if len(false_neg) > 0: plt.scatter(*zip(*false_neg), c="red", marker="o")
    if len(true_neg) > 0: plt.scatter(*zip(*true_neg), c="green", marker="x")
    if len(false_pos) > 0: plt.scatter(*zip(*false_pos), c="red", marker="x")

    if hyperplanes is not None:
        planes = [constraint_to_hyperplane(h) for conj in hyperplanes for h in conj if h.is_le() or h.is_lt()]
        for plane in planes:
            print(plane)
            if plane[0][feat_y] == 0:
                plt.plot([plane[1], plane[1]], [0, 1])
            else:
                plt.plot([0, 1], [(plane[1] - plane[0][feat_x] * x) / plane[0][feat_y] for x in [0, 1]])

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.savefig("{}.png".format(name))
    plt.gcf().clear()


def constraint_to_hyperplane(constraint):
    if constraint.is_le() or constraint.is_lt():
        coefficients = dict()
        left, right = constraint.args()
        if right.is_plus():
            left, right = right, left
        if left.is_plus():
            for term in left.args():
                if term.is_times():
                    c, v = term.args()
                    coefficients[v.symbol_name()] = float(c.constant_value())
                else:
                    raise RuntimeError("Unexpected value, expected product, was {}".format(term))
        else:
            raise RuntimeError("Unexpected value, expected sum, was {}".format(left))
        return coefficients, float(right.constant_value())
    raise RuntimeError("Unexpected constraint, expected inequality, was {}".format(constraint))


def find_border_points(domain, data):
    dist = []
    border_points = set()
    for i in range(len(data)):
        dist.append([])
        for j in range(len(data)):
            dist[i].append(get_distance(domain, data[i][0], data[j][0]))

    closest_others = []
    for i in range(len(data)):
        closest_other = None
        for j in range(len(data)):
            if data[j][1] != data[i][1]:
                if closest_other is None or dist[i][j] < dist[i][closest_other]:
                    closest_other = j
        closest_others.append(closest_other)

    for i in range(len(data)):
        if dist[i][closest_others[i]] == dist[closest_others[i]][closest_others[closest_others[i]]]:
            border_points.add(i)
            border_points.add(closest_others[i])

    return border_points


def find_border_points_fast(domain, data):
    positives = [i for i in range(len(data)) if data[i][1]]
    negatives = [i for i in range(len(data)) if not data[i][1]]

    closest = dict()
    for i in positives:
        for j in negatives:
            distance = get_distance(domain, data[i][0], data[j][0])
            if i not in closest or closest[i][1] > distance:
                closest[i] = (j, distance)
            if j not in closest or closest[j][1] > distance:
                closest[j] = (i, distance)

    return [i for i in range(len(data)) if closest[i][1] == closest[closest[i][0]][1]]


def get_distance(domain, example1, example2):
    distance = 0
    for v in domain.real_vars:
        distance += (example1[v] - example2[v]) ** 2
    return sqrt(distance)


def draw_border_points(feat_x, feat_y, data, border_indices, name):
    relevant_pos = []
    irrelevant_pos = []
    relevant_neg = []
    irrelevant_neg = []

    for i in range(len(data)):
        row = data[i]
        point = (row[0][feat_x], row[0][feat_y])
        if i in border_indices and row[1]:
            relevant_pos.append(point)
        elif i in border_indices and not row[1]:
            relevant_neg.append(point)
        elif i not in border_indices and row[1]:
            irrelevant_pos.append(point)
        else:
            irrelevant_neg.append(point)

    if len(relevant_pos) > 0: plt.scatter(*zip(*relevant_pos), c="green", marker="o")
    if len(irrelevant_pos) > 0: plt.scatter(*zip(*irrelevant_pos), c="grey", marker="o")
    if len(relevant_neg) > 0: plt.scatter(*zip(*relevant_neg), c="green", marker="x")
    if len(irrelevant_neg) > 0: plt.scatter(*zip(*irrelevant_neg), c="grey", marker="x")

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.savefig("{}.png".format(name))
    plt.gcf().clear()


def learn_parameter_free(problem, data, seed):
    def learn_inc(_data, _k, _h):
        learner = KCnfSmtLearner(_k, _h, RandomViolationsStrategy(20))
        dir_name = "../output/{}".format(problem.name)
        img_name = "{}_{}_{}_{}_{}".format(learner.name, _k, _h, len(data), seed)
        learner.add_observer(plotting.PlottingObserver(data, dir_name, img_name, "x", "y"))

        initial_indices = random.sample(list(range(len(data))), 20)

        learned_theory = learner.learn(problem.domain, data, initial_indices)
        # learned_theory = Or(*[And(*planes) for planes in hyperplane_dnf])
        print("Learned theory:\n{}".format(parse.smt_to_nested(learned_theory)))
        return learned_theory

    learn_bottom_up(data, learn_inc, 1, 1)


def main():
    n = 1000
    seed = 65
    random.seed(65)
    # problem = simple_univariate_problem()
    # problem = simple_checker_problem()
    # problem = simple_checker_problem_cnf()
    # problem = shared_hyperplane_problem()
    problem = checker_problem()
    # problem = cross_problem()
    # problem = bool_xor_problem()
    # data = [
    #     ({"x": Real(0.1), "y": Real(0.1)}, True),
    #     ({"x": Real(0.9), "y": Real(0.9)}, True),
    #     ({"x": Real(0.1), "y": Real(0.9)}, False),
    #     ({"x": Real(0.9), "y": Real(0.1)}, False),
    # ]

    sample_time_start = time.time()
    data = sample(problem, n, seed=seed)
    sample_time_elapsed = time.time() - sample_time_start
    print("Computing samples took {:.2f}s".format(sample_time_elapsed))

    learn_parameter_free(problem, data, seed)
    exit()

    start = time.time()
    border_indices = random.sample(list(range(len(data))), 20)
    # border_indices = find_border_points_fast(problem.domain, data)
    # border_finished_time = time.time() - start
    # print("Computing border points took {:.2f}s".format(border_finished_time))
    # border_indices = list(range(len(data)))
    border_points_name = "../output/{}_{}_{}".format(problem.name, len(data), seed)
    # draw_border_points("x", "y", data, border_indices, border_points_name)
    # data_subset = [data[i] for i in sorted(list(border_indices))]

    # draw_points("x", "y", data, "data")
    # learner = OCTLearner(1, 3, 0)
    # learner = KDNFLearner(2, 2, 0, negated_hyperplanes=True)
    # learner = KDNFLearner(2, 8, 0, negated_hyperplanes=False)
    # learner = GreedyLogicDNFLearner(2, 2)
    # learner = KDnfSmtLearner(8, 2)
    # learner = GreedyMaxRuleLearner(8)
    # learner = GreedyMilpRuleLearner(4, 4)
    learner = KCnfSmtLearner(2, 2, RandomViolationsStrategy(20))

    # if isinstance(learner, KDNFLearner):
    #     learner_name = "dnf"
    # elif isinstance(learner, GreedyLogicDNFLearner):
    #     learner_name = "dnf_logic_greedy"
    # elif isinstance(learner, KDNFLogicLearner):
    #     learner_name = "dnf_logic"
    # elif isinstance(learner, GreedyMaxRuleLearner):
    #     learner_name = "dnf_greedy_one_rule"
    # elif isinstance(learner, KDnfSmtLearner):
    #     learner_name = "dnf_smt"
    # else:
    #     learner_name = "dt"

    dir_name = "../output/{}".format(problem.name)
    img_name = "{}_{}_{}".format(learner.name, len(data), seed)
    learner.add_observer(plotting.PlottingObserver(data, dir_name, img_name, "x", "y"))

    domain = problem.domain
    learned_theory = learner.learn(domain, data, border_indices)
    # learned_theory = Or(*[And(*planes) for planes in hyperplane_dnf])
    print("Learned theory:\n{}".format(parse.smt_to_nested(learned_theory)))

    # learned_problem = Problem(domain, learned_theory, problem.name)
    # learned_labels = [(a, evaluate_assignment(learned_problem, a)) for a, _ in data]

    # img_name = "../output/{}_{}_{}_{}".format(learned_problem.name, learner.name, len(data), seed)

    # def filter_data(_data, _a):
    #     return [(_f, _label) for _f, _label in _data if all(_f[v] == Bool(_a[i]) for i, v in enumerate(domain.bool_vars))]

    # print(list(enumerate(domain.bool_vars)))

    # if len(domain.real_vars) == 2:
        # for bool_assignment in itertools.product([True, False], repeat=len(domain.bool_vars)):
            # bool_str = "_".join(("" if val else "not_") + var for var, val in zip(domain.bool_vars, bool_assignment))
            # filtered_learned = filter_data(learned_labels, bool_assignment)
            # filtered_data = filter_data(data, bool_assignment)
            # print("Actual labels")
            # print("\n".join("{}\tvs {}".format(filtered_data[i][1], filtered_learned[i][1]) for i in range(len(filtered_data))))
            # draw_points("x", "y", filtered_learned, img_name + bool_str, hyperplane_dnf, truth=filtered_data)
    # draw_points("x", "y", learned_labels, img_name + "_a", hyperplane_dnf, truth=data)
    # draw_points("x", "y", learned_labels, img_name + "_not_a", hyperplane_dnf, truth=data)


if __name__ == "__main__":
    main()
