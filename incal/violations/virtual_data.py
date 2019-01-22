from __future__ import print_function

import numpy as np
from pysmt.exceptions import InternalSolverError
from pysmt.environment import Environment
from pysmt.typing import REAL, BOOL

from .core import SelectionStrategy


class OneClassStrategy(SelectionStrategy):
    def __init__(self, regular_strategy, thresholds, tight_mode=True, class_label=True, background_knowledge=None):
        self.regular_strategy = regular_strategy  # type: SelectionStrategy
        self.thresholds = thresholds
        self.tight_mode = tight_mode
        assert class_label, "Currently only the positive setting is supported"
        self.class_label = class_label
        self.environment = Environment()
        if background_knowledge is None:
            self.background_knowledge = self.environment.formula_manager.TRUE()
        else:
            self.background_knowledge = self.environment.formula_manager.normalize(background_knowledge)

    def find_violating(self, domain, data, labels, formula):
        fm = self.environment.formula_manager
        formula = fm.normalize(formula)
        real_symbols = {name: fm.Symbol(name, REAL) for name in domain.real_vars}
        bool_symbols = {name: fm.Symbol(name, BOOL) for name in domain.bool_vars}
        symbols = real_symbols.copy()
        symbols.update(bool_symbols)
        bounds = domain.get_bounds(fm)
        try:
            with self.environment.factory.Solver() as solver:
                solver.add_assertion(formula)
                solver.add_assertion(bounds)
                solver.add_assertion(self.background_knowledge)
                # equalities = []
                # for row, label in data:
                #     for real_var in domain.real_vars:
                #         sym = real_symbols[real_var]
                #         val = fm.Real(row[real_var])
                #         t = fm.Real(self.thresholds[real_var])
                #         equalities.append(fm.Ite(sym >= val, fm.Equals(sym - val, t), fm.Equals(val - sym, t)))
                # solver.add_assertion(fm.Or(*equalities))
                for i in range(len(labels)):
                    row = {v: data[i, j].item() for j, v in enumerate(domain.variables)}
                    label = labels[i] == 1
                    if label == self.class_label:
                        constraint = fm.Implies(fm.And(
                            *[fm.Iff(bool_symbols[bool_var], fm.Bool(row[bool_var] == 1)) for bool_var in domain.bool_vars]),
                            fm.Or(*[fm.Ite(real_symbols[real_var] >= fm.Real(row[real_var]),
                                           real_symbols[real_var] - fm.Real(row[real_var]) >= fm.Real(
                                               self.thresholds[real_var]),
                                           fm.Real(row[real_var]) - real_symbols[real_var] >= fm.Real(
                                               self.thresholds[real_var])) for real_var in
                                    domain.real_vars]))
                    elif label == (not self.class_label):
                        constraint = fm.Implies(fm.And(
                            *[fm.Iff(bool_symbols[bool_var], fm.Bool(row[bool_var] == 1)) for bool_var in domain.bool_vars]),
                            fm.Or(*[fm.Ite(real_symbols[real_var] >= fm.Real(row[real_var]),
                                           real_symbols[real_var] - fm.Real(row[real_var]) >= fm.Real(
                                               self.thresholds[real_var]),
                                           fm.Real(row[real_var]) - real_symbols[real_var] >= fm.Real(
                                               self.thresholds[real_var])) for real_var in
                                    domain.real_vars]))
                    else:
                        raise ValueError("Unknown label l_{} = {}".format(i, label))
                    solver.add_assertion(constraint)
                solver.solve()
                model = solver.get_model()
                example = [float(model.get_value(symbols[var]).constant_value()) for var in domain.variables]
        except InternalSolverError:
            return None
        except Exception as e:
            if "Z3Exception" in str(type(e)):
                return None
            else:
                raise e

        return example

    def select_active(self, domain, data, labels, formula, active_indices):
        data, labels, selected = self.regular_strategy.select_active(domain, data, labels, formula, active_indices)
        if len(selected) > 0:
            return data, labels, selected
        else:
            example = self.find_violating(domain, data, labels, formula)
            if example is None:
                return data, labels, []
            data = np.vstack([data, example])
            labels = np.append(labels, np.array([0 if self.class_label else 1]))
            return data, labels, [len(labels) - 1]

"""
There is a point e, such that for every example e': d(e, e') > t
AND(OR(d(e, e', r) > t, forall r), forall e)
"""
