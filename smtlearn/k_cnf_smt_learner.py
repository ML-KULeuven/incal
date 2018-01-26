from __future__ import print_function

import pysmt.shortcuts as smt
from pysmt.typing import REAL

from incremental_learner import IncrementalLearner


class KCnfSmtLearner(IncrementalLearner):
    def __init__(self, conjunction_count, half_space_count, selection_strategy, allow_negations=True):
        IncrementalLearner.__init__(self, "cnf_smt", selection_strategy)
        self.conjunction_count = conjunction_count
        self.half_space_count = half_space_count
        self.allow_negations = allow_negations

    def learn_partial(self, solver, domain, data, new_active_indices):
        # Constants
        n_b_original = len(domain.bool_vars)
        n_b = n_b_original * 2
        n_r = len(domain.real_vars)

        n_h_original = self.half_space_count if n_r > 0 else 0
        n_h = n_h_original * 2 if self.allow_negations else n_h_original

        n_c = self.conjunction_count
        n_d = len(data)

        real_features = [[row[v] for v in domain.real_vars] for row, _ in data]
        bool_features = [[row[v] for v in domain.bool_vars] for row, _ in data]
        labels = [row[1] for row in data]

        # Variables
        a_hr = [[smt.Symbol("a_hr[{}][{}]".format(h, r), REAL) for r in range(n_r)] for h in range(n_h_original)]
        b_h = [smt.Symbol("b_h[{}]".format(h), REAL) for h in range(n_h_original)]
        s_ch = [[smt.Symbol("s_ch[{}][{}]".format(c, h)) for h in range(n_h)] for c in range(n_c)]
        s_cb = [[smt.Symbol("s_cb[{}][{}]".format(c, b)) for b in range(n_b)] for c in range(n_c)]

        # Aux variables
        s_ih = [[smt.Symbol("s_ih[{}][{}]".format(i, h)) for h in range(n_h)] for i in range(n_d)]
        s_ic = [[smt.Symbol("s_ic[{}][{}]".format(i, c)) for c in range(n_c)] for i in range(n_d)]

        # Constraints
        for i in new_active_indices:
            x_r, x_b, label = real_features[i], bool_features[i], labels[i]

            for h in range(n_h_original):
                sum_coefficients = smt.Plus([a_hr[h][r] * smt.Real(x_r[r]) for r in range(n_r)])
                solver.add_assertion(smt.Iff(s_ih[i][h], sum_coefficients <= b_h[h]))

            for h in range(n_h_original, n_h):
                solver.add_assertion(smt.Iff(s_ih[i][h], ~s_ih[i][h - n_h_original]))

            for c in range(n_c):
                solver.add_assertion(smt.Iff(s_ic[i][c], smt.Or(
                    [smt.FALSE()]
                    + [(s_ch[c][h] & s_ih[i][h]) for h in range(n_h)]
                    + [s_cb[c][b] for b in range(n_b_original) if x_b[b]]
                    + [s_cb[c][b] for b in range(n_b_original, n_b) if not x_b[b - n_b_original]]
                )))

            # for c in range(n_c):
            #     for h in range(n_h_original):
            #         solver.add_assertion(~(s_ch[c][h] & s_ch[c][h + n_h_original]))
            #     for b in range(n_b_original):
            #         solver.add_assertion(~(s_cb[c][b] & s_cb[c][b + n_b_original]))

            if label:
                solver.add_assertion(smt.And([s_ic[i][c] for c in range(n_c)]))
            else:
                solver.add_assertion(smt.Or([~s_ic[i][c] for c in range(n_c)]))

        solver.solve()
        model = solver.get_model()

        x_vars = [domain.get_symbol(domain.real_vars[r]) for r in range(n_r)]
        half_spaces = [
            smt.Plus([model.get_value(a_hr[h][r]) * x_vars[r] for r in range(n_r)]) <= model.get_value(b_h[h])
            for h in range(n_h_original)
        ] + [
            smt.Plus([model.get_value(a_hr[h][r]) * x_vars[r] for r in range(n_r)]) > model.get_value(b_h[h])
            for h in range(n_h - n_h_original)
        ]

        b_vars = [domain.get_symbol(domain.bool_vars[b]) for b in range(n_b_original)]
        bool_literals = [b_vars[b] for b in range(n_b_original)]
        bool_literals += [~b_vars[b] for b in range(n_b - n_b_original)]

        conjunctions = [
            [half_spaces[h] for h in range(n_h) if model.get_py_value(s_ch[c][h])]
            + [bool_literals[b] for b in range(n_b) if model.get_py_value(s_cb[c][b])]
            for c in range(n_c)
        ]

        return smt.And([smt.Or(conjunction) for conjunction in conjunctions])
