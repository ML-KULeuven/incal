import random

from pysmt.shortcuts import *

from dt_learner import OCTLearner


class Domain(object):
    def __init__(self, variables, var_types, var_domains):
        self.variables = variables
        self.var_types = var_types
        self.var_domains = var_domains

    def get_symbol(self, variable):
        return Symbol(variable, self.var_types[variable])


class Problem(object):
    def __init__(self, domain, theory):
        self.domain = domain
        self.theory = theory


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

    return Problem(Domain(variables, var_types, var_domains), theory)


def simple_univariate_problem():
    variables = ["x"]
    var_types = {"x": REAL}
    var_domains = {"x": (0, 1)}

    theory = LE(Symbol("x", REAL), Real(0.6))

    return Problem(Domain(variables, var_types, var_domains), theory)


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


def main():
    problem = checker_problem()
    data = sample(problem, 100, seed=65)
    learner = OCTLearner(1, 3, 0.2)
    learned_theory = learner.learn(problem.domain, data)
    # print(serialize(learned_theory))


if __name__ == "__main__":
    main()