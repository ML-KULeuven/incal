import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

from unittest import TestCase

from pysmt.typing import REAL
import pysmt.shortcuts as smt

from problem import Domain
from visualize import RegionBuilder


class TestPolytope(TestCase):
    def test_example1(self):
        domain = Domain(["x", "y"], {"x": REAL, "y": REAL}, {"x": [0, 1], "y": [0, 1]})
        x, y = smt.Symbol("x", REAL), smt.Symbol("y", REAL)
        formula = (x + y <= 0.5)
        RegionBuilder(domain).walk_smt(formula).plot()
        plt.show()


