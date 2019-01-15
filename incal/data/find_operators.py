import pywmi
from pywmi import SmtWalker


class OperatorWalker(SmtWalker):
    def __init__(self):
        self.operators = set()

    def walk_and(self, args):
        self.operators.add("&")
        self.walk_smt_multiple(args)

    def walk_or(self, args):
        self.operators.add("|")
        self.walk_smt_multiple(args)

    def walk_plus(self, args):
        self.operators.add("+")
        self.walk_smt_multiple(args)

    def walk_minus(self, left, right):
        self.operators.add("-")
        self.walk_smt_multiple([left, right])

    def walk_times(self, args):
        self.operators.add("*")
        self.walk_smt_multiple(args)

    def walk_not(self, argument):
        self.operators.add("~")
        self.walk_smt_multiple([argument])

    def walk_ite(self, if_arg, then_arg, else_arg):
        self.operators.add("ite")
        self.walk_smt_multiple([if_arg, then_arg, else_arg])

    def walk_pow(self, base, exponent):
        self.operators.add("^")
        self.walk_smt_multiple([base, exponent])

    def walk_lte(self, left, right):
        self.operators.add("<=")
        self.walk_smt_multiple([left, right])

    def walk_lt(self, left, right):
        self.operators.add("<")
        self.walk_smt_multiple([left, right])

    def walk_equals(self, left, right):
        self.operators.add("=")
        self.walk_smt_multiple([left, right])

    def walk_symbol(self, name, v_type):
        pass

    def walk_constant(self, value, v_type):
        pass

    def find_operators(self, formula):
        self.walk_smt(formula)
        return list(self.operators)
