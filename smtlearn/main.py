import random

from pysmt.shortcuts import *
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


from dt_learner import OCTLearner
from k_dnf_learner import KDNFLearner


class Domain(object):
    def __init__(self, variables, var_types, var_domains):
        self.variables = variables
        self.var_types = var_types
        self.var_domains = var_domains

    def get_symbol(self, variable):
        return Symbol(variable, self.var_types[variable])


class Problem(object):
    def __init__(self, domain, theory, name):
        self.domain = domain
        self.theory = theory
        self.name = name


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


def evaluate_assignment(problem, assignment):
    substitution = {problem.domain.get_symbol(v): val for v, val in assignment.items()}
    return problem.theory.substitute(substitution).simplify().is_true()


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
    generator = random.Random(seed)
    for i in range(n):
        instance = dict()
        for v in problem.domain.variables:
            if problem.domain.var_types[v] == REAL:
                lb, ub = problem.domain.var_domains[v]
                instance[v] = Real(generator.uniform(lb, ub))
            elif problem.domain.var_types[v] == BOOL:
                instance[v] = Bool(True if generator.random() < 0.5 else False)
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
        planes = [constraint_to_hyperplane(h) for conj in hyperplanes for h in conj]
        for plane in planes:
            print(plane)
            plt.plot([0, 1], [(plane[1] - plane[0][feat_x] * x) / plane[0][feat_y] for x in [0, 1]])

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.savefig("{}.png".format(name))


def constraint_to_hyperplane(constraint):
    if constraint.is_le():
        coefficients = dict()
        left, right = constraint.args()
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


def main():
    n = 200
    problem = simple_univariate_problem()
    problem = simple_checker_problem()
    problem = shared_hyperplane_problem()
    data = [
        ({"x": Real(0), "y": Real(0)}, True),
        ({"x": Real(1), "y": Real(1)}, True),
        ({"x": Real(0), "y": Real(1)}, False),
        ({"x": Real(1), "y": Real(0)}, False),
    ]
    data = sample(problem, n, seed=65)
    # draw_points("x", "y", data, "data")
    # learner = OCTLearner(1, 3, 0)
    learner = KDNFLearner(2, 6, 0)

    learner_name = "dnf" if isinstance(learner, KDNFLearner) else "dt"

    hyperplane_dnf = learner.learn(problem.domain, data)
    learned_theory = Or(*[And(*planes) for planes in hyperplane_dnf])
    print(serialize(learned_theory))

    learned_problem = Problem(problem.domain, learned_theory, problem.name)
    learned_labels = [(a, evaluate_assignment(learned_problem, a)) for a, _ in data]
    img_name = "{}_{}".format(learned_problem.name, learner_name)
    draw_points("x", "y", learned_labels, img_name, hyperplane_dnf, truth=data)


if __name__ == "__main__":
    main()
