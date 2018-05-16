from __future__ import print_function
import numpy
from pysmt.typing import REAL

from smt_walk import SmtWalker
import polytope as pc


class RegionBuilder(SmtWalker):
    def __init__(self, domain):
        self.domain = domain

    def get_bounded_region(self, in_c, in_b):
        coefficients = numpy.zeros([len(self.domain.real_vars) * 2 + 1, len(self.domain.real_vars)])
        b = numpy.zeros(len(self.domain.real_vars) * 2 + 1)
        for i in range(len(self.domain.real_vars)):
            coefficients[2 * i, i] = -1
            coefficients[2 * i + 1, i] = 1

            lb, ub = self.domain.var_domains[self.domain.real_vars[i]]
            b[2 * i] = -lb
            b[2 * i + 1] = ub

        coefficients[-1, :] = in_c
        b[-1] = in_b

        return pc.Region([pc.Polytope(coefficients, b)])

    def walk_and(self, args):
        regions = self.walk_smt_multiple(args)
        region = regions[0]
        for i in range(1, len(regions)):
            region = region.intersect(regions[i])
        return region

    def walk_or(self, args):
        regions = self.walk_smt_multiple(args)
        region = regions[0]
        for i in range(1, len(regions)):
            region = region.union(regions[i])
        return region

    def walk_plus(self, args):
        coefficients = dict()
        for arg in self.walk_smt_multiple(args):
            if not isinstance(arg, dict) and isinstance(arg, str):
                arg = {arg: 1}
            coefficients.update(arg)
        return coefficients

    def walk_minus(self, left, right):
        raise RuntimeError("Should not encounter minus")

    def walk_times(self, args):
        args = self.walk_smt_multiple(args)
        if len(args) != 2:
            raise RuntimeError("Something went wrong, expected 2 arguments but got {}".format(args))
        if isinstance(args[0], str):
            return {args[0]: args[1]}
        else:
            return {args[1]: args[0]}

    def walk_not(self, argument):
        return self.get_bounded_region(numpy.zeros(len(self.domain.real_vars)), 0) - self.walk_smt(argument)

    def walk_ite(self, if_arg, then_arg, else_arg):
        raise RuntimeError("Should not encounter ite")

    def walk_pow(self, base, exponent):
        raise RuntimeError("Should not encounter power")

    def walk_lte(self, left, right):
        left, right = self.walk_smt_multiple((left, right))
        if isinstance(right, dict):
            t = right
            right = left
            left = t
            right = -right
            left = {v: -val for v, val in left.items()}

        coefficients = numpy.array([left[v] for v in self.domain.real_vars])

        inequality = self.get_bounded_region(coefficients, right)
        return inequality

    def walk_lt(self, left, right):
        return self.walk_lte(left, right)

    def walk_equals(self, left, right):
        raise RuntimeError("Should not encounter equals")

    def walk_symbol(self, name, v_type):
        return name

    def walk_constant(self, value, v_type):
        if v_type == REAL:
            return float(value)

        whole = self.get_bounded_region(numpy.zeros(len(self.domain.real_vars)), 0)
        if value:
            return whole
        else:
            return whole - whole
