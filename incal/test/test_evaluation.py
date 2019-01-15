import numpy as np

from smtlearn.examples import ice_cream_problem
from pywmi import Domain, evaluate


def test_example1():
    domain, formula, name = ice_cream_problem()
    c, b, w = domain.get_symbols(["chocolate", "banana", "weekend"])

    c_val = 0.41358769878652346
    b_val = 0.04881279380000003
    assignment = {"chocolate": c_val, "banana": b_val, "weekend": 1.0}
    instance = np.array([assignment[v] for v in domain.variables])

    h1 = -0.9094061613514598 < (-2.11558444119424*c + -0.7052753601938021*b)
    print(-0.9094061613514598, (-2.11558444119424 * c_val + -0.7052753601938021 * b_val))
    h2 = -43.62318633585081 < (-56.41097694745345*c + -50.5657977670196*b)
    print(-43.62318633585081, (-56.41097694745345 * c_val + -50.5657977670196 * b_val))
    h3 = -0.9094061613514598 < (-2.11558444119424*c + -0.7052753601938021*b)
    print(-0.9094061613514598, (-2.11558444119424 * c_val + -0.7052753601938021 * b_val))
    h4 = 7.792607696237757 < (18.128225098004087*c + 6.043431893671825*b)
    print(7.792607696237757, (18.128225098004087 * c_val + 6.043431893671825 * b_val))
    h5 = -0.9094061613514598 < -(2.11558444119424*c + -0.7052753601938021*b)
    print(-0.9094061613514598, -(2.11558444119424 * c_val + -0.7052753601938021 * b_val))
    # h1: True, h2: True, h3: True, h4: False, h5: True

    learned = ((h1 | h2) & (h3 | ~w) & (h4 | h5))

    print(evaluate(domain, formula, instance))
    print(evaluate(domain, learned, instance))


test_example1()