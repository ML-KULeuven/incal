from pywmi import Domain

from lp.model import Model


def lp_domain(n, ranges=None):
    if ranges is None:
        ranges = [(None, None) for i in range(n)]
    return Domain.make([], ["x{}".format(i + 1) for i in range(n)], ranges)


def lp_2_6() -> Model:
    domain = lp_domain(2)
    x1, x2 = domain.get_symbols(domain.variables)
    return Model(
        domain,
        300 * x1 + 200 * x2,
        [
            2 * x1 + x2 <= 100,
            x1 + x2 <= 80,
            x1 <= 40,
            x1 >= 0,
            x2 >= 0
        ],
        minimize=False,
        name="LP_2_6"
    )


def lp_2_7() -> Model:
    domain = lp_domain(4)
    x1, x2, x3, x4 = domain.get_symbols(domain.variables)
    return Model(
        domain,
        320 * x1 + 400 * x2 + 480 * x3 + 560 * x4,
        [
            0.06 * x1 + 0.03 * x2 + 0.02 * x3 + 0.01 * x4 >= 3.5,
            0.03 * x1 + 0.02 * x2 + 0.05 * x3 + 0.06 * x4 <= 3,
            0.08 * x1 + 0.03 * x2 + 0.02 * x3 + 0.01 * x4 == 4,
            x1 + x2 + x3 + x4 == 110,
        ] + [x >= 0 for x in domain.get_symbols(domain.variables)],
        minimize=False,
        name="LP_2_7"
    )


def lp_2_8() -> Model:
    domain = lp_domain(3)
    x1, x2, x3 = domain.get_symbols(domain.variables)
    return Model(
        domain,
        5 * x1 + 4 * x2 + 3 * x3,
        [
            2 * x1 + 3 * x2 + x3 <= 5,
            4 * x1 + x2 + 2 * x3 <= 11,
            3 * x1 + 4 * x2 + 2 * x3 <= 5,
        ] + [x >= 0 for x in domain.get_symbols(domain.variables)],
        minimize=False,
        name="LP_2_8"
    )


def lp_2_9() -> Model:
    domain = lp_domain(4)
    x1, x2, x3, x4 = domain.get_symbols(domain.variables)
    return Model(
        domain,
        3 * x1 - x2,
        [
            0 - x1 + 6 * x2 - x3 + x4 >= -3,
            7 * x2 + 2 * x4 == 5,
            x1 + x2 + x3 - x4 <= 2,
            x1 >= 0,
            x3 >= 0,
        ],
        minimize=True,
        name="LP_2_9"
    )


