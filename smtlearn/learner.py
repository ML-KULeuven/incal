import numpy as np
import pysmt.shortcuts as smt


class Learner(object):
    def __init__(self, name):
        self.name = name

    def learn(self, domain, data, border_indices):
        raise NotImplementedError()

    @staticmethod
    def _convert(value):
        return float(value.constant_value())

    @staticmethod
    def _get_misclassification(data):
        true_count = 0
        for _, l in data:
            if l:
                true_count += 1
        return min(true_count, len(data) - true_count)

    @staticmethod
    def check_example(domain, example_features, dnf_list):
        x_vars = [domain.get_symbol(var) for var in domain.real_vars]
        b_vars = [domain.get_symbol(var) for var in domain.bool_vars]

        formula = smt.Or([smt.And(hyperplane_conjunct) for hyperplane_conjunct in dnf_list])
        substitution = {var: example_features[str(var)] for var in x_vars + b_vars}
        return formula.substitute(substitution).simplify().is_true()

    @staticmethod
    def fit_hyperplane(domain, examples):
        matrix = np.matrix([[example[v] for v in domain.real_vars] for example in examples])
        k = np.ones((len(examples), 1))
        a = np.matrix.dot(np.linalg.inv(matrix), k)
        return a, 1
