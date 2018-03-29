from pysmt.environment import Environment
from pysmt.typing import REAL, BOOL

from incremental_learner import SelectionStrategy


class OneClassStrategy(SelectionStrategy):
    def __init__(self, regular_strategy, thresholds, tight_mode=True, class_label=True):
        self.regular_strategy = regular_strategy
        self.thresholds = thresholds
        self.tight_mode = tight_mode
        assert class_label, "Currently only the positive setting is supported"
        self.class_label = class_label
        self.environment = Environment()

    def find_violating(self, domain, data, formula):
        fm = self.environment.formula_manager
        formula = fm.normalize(formula)
        real_symbols = {name: fm.Symbol(name, REAL) for name in domain.real_vars}
        bool_symbols = {name: fm.Symbol(name, BOOL) for name in domain.bool_vars}
        symbols = real_symbols.copy()
        symbols.update(bool_symbols)
        with self.environment.factory.Solver() as solver:
            solver.add_assertion(formula)
            for row, label in data:
                if label == self.class_label:
                    print(len(data), label, self.class_label, self.thresholds, domain.real_vars)
                    constraint = fm.Implies(fm.And(
                        *[fm.Iff(bool_symbols[bool_var], fm.Bool(row[bool_var])) for bool_var in domain.bool_vars]),
                                         fm.Or(*[fm.Ite(real_symbols[real_var] >= fm.Real(row[real_var]),
                                                        real_symbols[real_var] - fm.Real(row[real_var]) > fm.Real(
                                                            self.thresholds[real_var]),
                                                        fm.Real(row[real_var]) - real_symbols[real_var] > fm.Real(
                                                            self.thresholds[real_var])) for real_var in
                                                 domain.real_vars]))
                    print(self.environment.serializer.serialize(constraint))
                    solver.add_assertion(constraint)
            solver.solve()
            model = solver.get_model()
            example = {var: model.get_value(symbols[var]).constant_value() for var in domain.variables}
        return example

    def select_active(self, domain, data, formula, active_indices):
        regular_selection = self.regular_strategy.select_active(domain, data, formula, active_indices)
        if len(regular_selection) > 0:
            return regular_selection
        else:
            example = self.find_violating(domain, data, formula)
            data.append((example, not self.class_label))
            print(example)
            return [len(data) - 1]

"""
There is a point e, such that for every example e': d(e, e') > t
AND(OR(d(e, e', r) > t, forall r), forall e)
"""