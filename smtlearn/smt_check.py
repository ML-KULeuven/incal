import pysmt.shortcuts as smt

from smt_walk import SmtWalker


class SmtChecker(SmtWalker):
    def __init__(self, assignment):
        self.assignment = assignment

    def walk_plus(self, args):
        return sum(self.walk_smt_multiple(args))

    def walk_minus(self, left, right):
        return self.walk_smt(left) - self.walk_smt(right)

    def walk_lt(self, left, right):
        return self.walk_smt(left) < self.walk_smt(right)

    def walk_ite(self, if_arg, then_arg, else_arg):
        return self.walk_smt(then_arg) if self.walk_smt(if_arg) else self.walk_smt(else_arg)

    def walk_and(self, args):
        return all(self.walk_smt_multiple(args))

    def walk_symbol(self, name, v_type):
        return self.walk_constant(self.assignment[name], v_type)

    def walk_constant(self, value, v_type):
        if v_type == smt.BOOL:
            if isinstance(value, bool):
                return bool(value)
            return bool(value.constant_value())
        elif v_type == smt.REAL:
            try:
                return float(value)
            except TypeError:
                return float(value.constant_value())
        raise RuntimeError("Unsupported type {}".format(v_type))

    def walk_lte(self, left, right):
        return self.walk_smt(left) <= self.walk_smt(right)

    def walk_equals(self, left, right):
        return self.walk_smt(left) == self.walk_smt(right)

    def walk_or(self, args):
        return any(self.walk_smt_multiple(args))

    def walk_pow(self, base, exponent):
        return base ** exponent

    def walk_times(self, args):
        if len(args) > 0:
            aggregate = 1
            for res in self.walk_smt_multiple(args):
                aggregate *= res
            return aggregate
        raise RuntimeError("Zero argument multiplication")

    def walk_not(self, argument):
        return ~self.walk_smt(argument)

    def check(self, formula):
        return self.walk_smt(formula)
