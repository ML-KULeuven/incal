import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

from pysmt.typing import REAL
import pysmt.shortcuts as smt

from problem import Domain
from visualize import RegionBuilder


def xy_domain():
    return Domain(["x", "y"], {"x": REAL, "y": REAL}, {"x": [0, 1], "y": [0, 1]})


def example1(domain):
    x, y = smt.Symbol("x", REAL), smt.Symbol("y", REAL)
    return domain, (x + y <= 0.5)


def example2(domain):
    x, y = smt.Symbol("x", REAL), smt.Symbol("y", REAL)
    return domain, (((-1.81491574069 < 2.82223533496 * x + -2.86421413834 * y) | (
                1.74295350642 < 5.75692214636 * x + -5.67797696689 * y)) & (
                 5.75692214636 * x + -5.67797696689 * y <= 1.74295350642))


def example3(domain):
    x, y = smt.Symbol("x", REAL), smt.Symbol("y", REAL)
    return domain, (((5.03100425089 < 4.72202520763*x + 4.11473198213*y) | (-4.6261635019 < -5.93640712709*x + -5.87100650773*y)) & ((5.03100425089 < 4.72202520763*x + 4.11473198213*y) | (-4.6261635019 < -5.93640712709*x + -5.87100650773*y)))


def example4(domain):
    x, y = smt.Symbol("x", REAL), smt.Symbol("y", REAL)
    return domain, (((106.452209182 < 58.3305562428*x + 162.172448357*y) | (-82.1173457701 < -121.782718841*x + -45.7311195244*y)) & ((58.3305562428*x + 162.172448357*y <= 106.452209182) | (-121.782718841*x + -45.7311195244*y <= -82.1173457701)))


def example5(domain):
    x, y = smt.Symbol("x", REAL), smt.Symbol("y", REAL)
    return domain, (((-1.81491574069 < 2.82223533496*x + -2.86421413834*y) | (1.74295350642 < 5.75692214636*x + -5.67797696689*y)) & (5.75692214636*x + -5.67797696689*y <= 1.74295350642))


def example6(domain):
    x, y = smt.Symbol("x", REAL), smt.Symbol("y", REAL)
    return domain, (((-1.27554738321 < 2.00504448571*x + -2.40276942762*y) | (4.56336137649 < 11.0066321223*x + -9.72098326672*y)) & (11.0066321223*x + -9.72098326672*y <= 4.56336137649))


def visualize(domain, formula):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    RegionBuilder(domain).walk_smt(formula).plot(ax=ax)
    plt.show()


if __name__ == "__main__":
    visualize(*example6(xy_domain()))
