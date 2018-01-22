from __future__ import print_function

import itertools

import numpy as np
import pysmt.shortcuts as smt
import time
from pysmt.typing import REAL, BOOL

from learner import Learner


class KDnfSmtLearner(Learner):
    def __init__(self, max_hyperplanes, max_terms, allow_negations=True):
        Learner.__init__(self)
        self.max_hyperplanes = max_hyperplanes
        self.max_terms = max_terms
        self.allow_negations = allow_negations

    def learn(self, domain, data, initial_indices=None):
        # Constants
        n_b_original = len(domain.bool_vars)
        n_b = n_b_original * 2 if self.allow_negations else n_b_original

        n_f = len(domain.real_vars)
        n_h_original = self.max_hyperplanes if n_f > 0 else 0
        n_h = n_h_original * 2 if self.allow_negations else n_h_original

        n_c = self.max_terms
        n_d = len(data)

        real_features = [[Learner._convert(row[v]) for v in domain.real_vars] for row, _ in data]
        bool_features = [[bool(row[v].constant_value()) for v in domain.bool_vars] for row, _ in data]
        labels = [row[1] for row in data]

        # Variables
        a_hf = [[smt.Symbol("a_hf[{}][{}]".format(h, f), REAL) for f in range(n_f)] for h in range(n_h_original)]
        b_h = [smt.Symbol("b_h[{}]".format(h), REAL) for h in range(n_h_original)]
        s_ch = [[smt.Symbol("s_ch[{}][{}]".format(c, h)) for h in range(n_h)] for c in range(n_c)]
        s_cb = [[smt.Symbol("s_cb[{}][{}]".format(c, b)) for b in range(n_b)] for c in range(n_c)]

        # Aux variables
        s_ih = [[smt.Symbol("s_ih[{}][{}]".format(i, h)) for h in range(n_h)] for i in range(n_d)]
        s_ic = [[smt.Symbol("s_ic[{}][{}]".format(i, c)) for c in range(n_c)] for i in range(n_d)]

        # Constraints
        start = time.time()
        active_indices = list(range(len(data))) if initial_indices is None else initial_indices
        remaining = list(range(len(data)))  # list(sorted(set(range(len(data))) - set(active_indices)))

        hyperplane_dnf = []

        def check_model(_x):
            _formula = smt.Or([smt.And(hyperplane_conjunct) for hyperplane_conjunct in hyperplane_dnf])
            substitution = {_var: _x[str(_var)] for _var in x_vars + b_vars}
            return _formula.substitute(substitution).simplify().is_true()

        print("Starting solver with {} examples".format(len(active_indices)))

        with smt.Solver() as solver:
            while len(active_indices) > 0:
                remaining = list(sorted(set(remaining) - set(active_indices)))
                for i in active_indices:
                    x, x_b, label = real_features[i], bool_features[i], labels[i]

                    for h in range(n_h_original):
                        sum_coefficients = smt.Plus([a_hf[h][f] * smt.Real(x[f]) for f in range(n_f)])
                        solver.add_assertion(smt.Iff(s_ih[i][h], sum_coefficients <= b_h[h]))

                    for h in range(n_h_original, n_h):
                        solver.add_assertion(smt.Iff(s_ih[i][h], ~s_ih[i][h - n_h_original]))

                    for c in range(n_c):
                        solver.add_assertion(smt.Iff(s_ic[i][c], smt.And(
                            [(~s_ch[c][h] | s_ih[i][h]) for h in range(n_h)]
                            + [~s_cb[c][b] for b in range(n_b_original) if not x_b[b]]
                            + [~s_cb[c][b] for b in range(n_b_original, n_b) if x_b[b - n_b_original]]
                        )))

                    if label:
                        solver.add_assertion(smt.Or([s_ic[i][c] for c in range(n_c)]))
                    else:
                        solver.add_assertion(smt.And([~s_ic[i][c] for c in range(n_c)]))

                solver.solve()
                model = solver.get_model()

                x_vars = [domain.get_symbol(domain.variables[f]) for f in range(n_f)]
                hyperplanes = [
                    smt.Plus([model.get_value(a_hf[h][f]) * x_vars[f] for f in range(n_f)]) <= model.get_value(b_h[h])
                    for h in range(n_h_original)]
                hyperplanes += [
                    smt.Plus([model.get_value(a_hf[h][f]) * x_vars[f] for f in range(n_f)]) > model.get_value(b_h[h])
                    for h in range(n_h - n_h_original)]

                b_vars = [domain.get_symbol(domain.bool_vars[b]) for b in range(n_b_original)]
                bool_literals = [b_vars[b] for b in range(n_b_original)]
                bool_literals += [~b_vars[b - n_b_original] for b in range(n_b_original, n_b)]

                hyperplane_dnf = [
                    [hyperplanes[h] for h in range(n_h) if model.get_py_value(s_ch[c][h])]
                    + [bool_literals[b] for b in range(n_b) if model.get_py_value(s_cb[c][b])]
                    for c in range(n_c)
                ]

                active_indices = [i for i in remaining if labels[i] != check_model(data[i][0])]
                print("Found model violating {} examples".format(len(active_indices)))

        time_taken = time.time() - start
        print("Took {:.2f}s".format(time_taken))
        return hyperplane_dnf
