import numpy as np

from incal.generator import generate_half_space_sample
from incal.learner import Learner
from pysmt.typing import REAL, BOOL
from pywmi import Domain


def get_xay_domain():
    return Domain(["x", "a", "y"], {"x": REAL, "a": BOOL, "y": REAL}, {"x": (0, 1), "y": (0, 1)})


def test_generate_hyperplane():
    domain = get_xay_domain()
    samples = np.array([[0, 1, 0.1], [0.5, 0, 0.5]])
    coefficients, b = Learner.fit_hyperplane(domain, samples)
    slope = coefficients[0] / coefficients[1]
    assert abs(slope) == 0.4 / 0.5
    assert b == 1


def test_generate_hyperplane_sample_sanity():
    generate_half_space_sample(get_xay_domain(), 2)
