from pywmi import Domain
from pysmt.shortcuts import REAL, Or, And, LE, Real, Symbol, BOOL, GT, Not, Plus, Times, GE


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

    return xy_domain(), theory, "simple_checker"


def simple_checker_problem_cnf():
    x, y = (Symbol(n, REAL) for n in ["x", "y"])
    theory = ((x <= 0.5) | (y > 0.5)) & ((x > 0.5) | (y <= 0.5))
    return xy_domain(), theory, "simple_cnf_checker"


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

    return Domain(variables, var_types, var_domains), theory, "checker"


def simple_univariate_problem():
    variables = ["x"]
    var_types = {"x": REAL}
    var_domains = {"x": (0, 1)}

    theory = LE(Symbol("x", REAL), Real(0.6))

    return Domain(variables, var_types, var_domains), theory, "one_test"


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
    return domain, Or(And(shared1, shared2, h1, h2), And(shared1, shared2, h3, h4)), "shared"


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
    return domain, theory, "cross"


def bool_xor_problem():
    variables = ["a", "b"]
    var_types = {"a": BOOL, "b": BOOL}
    var_domains = dict()
    domain = Domain(variables, var_types, var_domains)

    a, b = (domain.get_symbol(v) for v in variables)

    theory = (a & ~b) | (~a & b)
    return domain, theory, "2xor"


def ice_cream_problem():
    variables = ["chocolate", "banana", "weekend"]
    chocolate, banana, weekend = variables
    var_types = {chocolate: REAL, banana: REAL, weekend: BOOL}
    var_domains = {chocolate: (0, 1), banana: (0, 1)}
    domain = Domain(variables, var_types, var_domains)

    chocolate, banana, weekend = (domain.get_symbol(v) for v in variables)
    theory = (chocolate < 0.650) \
             & (banana < 0.550) \
             & (chocolate + 0.7 * banana <= 0.700) \
             & (chocolate + 1.2 * banana <= 0.750) \
             & (~weekend | (chocolate + 0.7 * banana <= 0.340))

    return domain, theory, "ice_cream"


def get_all():
    return [
        simple_checker_problem(),
        simple_checker_problem_cnf(),
        checker_problem(),
        simple_univariate_problem(),
        shared_hyperplane_problem(),
        cross_problem(),
        bool_xor_problem(),
        ice_cream_problem(),
    ]


def get_by_name(name):
    for t in get_all():
        if t[2] == name:
            return t[0], t[1]
