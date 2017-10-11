from __future__ import print_function

from gurobipy.gurobipy import *


# noinspection PyPep8Naming
from learner import Learner


class KDNFLearner(Learner):
    def __init__(self, k, max_hyperplanes, alpha, min_covered=None, max_terms=None):
        self.k = k
        self.max_hyperplanes = max_hyperplanes
        self.alpha = alpha
        self.min_covered = min_covered
        self.max_terms = max_terms
        self.mu = 0.0005

    def learn(self, domain, data):
        # Create a new model
        m = Model("kDNF")

        print("Data")
        for row, l in data:
            print(*(["{}: {:.2f}".format(v, Learner._convert(row[v])) for v in domain.variables] + [l]))
        print()

        # Computed constants
        n_f = len(domain.variables)
        n_h = self.max_hyperplanes
        n_d = len(data)
        k = self.k
        naive_misclassification = float(Learner._get_misclassification(data))

        x = [[Learner._convert(row[v]) for v in domain.variables] for row, _ in data]


        # Variables
        s_h = [m.addVar(vtype=GRB.BINARY, name="s_H({h})".format(h=h)) for h in range(n_h)]
        s_d = [m.addVar(vtype=GRB.BINARY, name="s_D({i})".format(i=i)) for i in range(n_d)]

        z_h = [
            [m.addVar(vtype=GRB.BINARY, name="z_H({h}, {c})".format(h=h, c=c)) for c in range(k)]
            for h in range(n_h)
        ]
        z_d = [
            [m.addVar(vtype=GRB.BINARY, name="z_D({i}, {c})".format(i=i, c=c)) for c in range(k)]
            for i in range(n_d)
        ]
        z_ih = [
            [m.addVar(vtype=GRB.BINARY, name="z_IH({i}, {h})".format(i=i, h=h)) for h in range(n_h)]
            for i in range(n_d)
        ]
        z_ihc = [
            [
                [m.addVar(vtype=GRB.BINARY, name="z_IHC({i}, {h}, {c})".format(i=i, h=h, c=c)) for c in range(k)]
                for h in range(n_h)
            ]
            for i in range(n_d)
        ]

        a = [
            [m.addVar(vtype=GRB.CONTINUOUS, lb=-1, ub=1, name="a({h}, {j})".format(h=h, j=j)) for j in range(n_f)]
            for h in range(n_h)
        ]
        b = [m.addVar(vtype=GRB.CONTINUOUS, lb=-1, ub=1, name="b({h})".format(h=h)) for h in range(n_h)]
        print()

        # Set objective
        misclassification = sum(1 - s_d[i] if data[i][1] else s_d[i] for i in range(n_d))
        m.setObjective(misclassification / naive_misclassification + self.alpha * sum(s_h), GRB.MINIMIZE)

        # Add constraint: SUM_(j = 1..n_h) a(h, j) * x(i, j) <= b(h) + M(1 - z_ih(i, h)) for all h, i
        for h in range(n_h):
            for i in range(n_d):
                name = "SUM_(j = 1..n_H) a({h}, j) * x({i}, j) <= b({h}) + M(1 - z_IH({i}, {h}))".format(h=h, i=i)
                m.addConstr(sum(a[h][j] * x[i][j] for j in range(n_f)) <= b[h] + (n_f + 1) * (1 - z_ih[i][h]), name)
                print(name)
        print()

        for h in range(n_h):
            for i in range(n_d):
                name = "SUM_(j = 1..n_H) a({h}, j) * x({i}, j) >= b({h}) - mu - M * z_IH({i}, {h})".format(h=h, i=i)
                m.addConstr(sum(a[h][j] * x[i][j] for j in range(n_f)) >= b[h] + self.mu - (n_f + 1) * z_ih[i][h], name)
                print(name)
        print()

        # If a hyperplane is assigned to any conjunction, it is selected (used)
        for h in range(n_h):
            for c in range(k):
                name = "s_H({h}) >= z_H({h}, {c})".format(h=h, c=c)
                m.addConstr(s_h[h] >= z_h[h][c], name)
                print(name)
        print()

        # If a hyperplane is not assigned to any conjunction, it cannot be selected (used)
        for h in range(n_h):
            name = "s_H({h}) <= SUM_(c = 1..k) z_H({h}, c)".format(h=h)
            m.addConstr(s_h[h] <= sum(z_h[h][c] for c in range(k)), name)
            print(name)
        print()

        # If an example i is not assigned to c, then it cannot be assigned to every h that is assigned to c
        # for c in range(k):
        #     for i in range(n_d):
        #         # TODO name
        #         name = "z_IH({i}, {h}) + z_H({h}, {c}) >= 2 - 2 * (1 - z_D({i}, {c}))".format(h=h, c=c, i=i)
        #         m.addConstr(sum(z_h[h][c] for h in range(n_h)) - sum(z_ihc[i][h][c] for h in range(n_h)) <= n_h * (1 - z_d[i][c]))
        #         m.addConstr(sum(z_h[h][c] for h in range(n_h)) - sum(z_ihc[i][h][c] for h in range(n_h)) >= n_h * (1 - z_d[i][c]))
        #         print(name)
        # print()

        # If (h, c) then (i, c) can only be selected if every (i, h, c) is selected
        for c in range(k):
            for h in range(n_h):
                for i in range(n_d):
                    name = "z_D({i}, {c}) <= z_IHC({i}, {h}, {c}) + (1 - z_H({h}, {c}))".format(c=c, h=h, i=i)
                    m.addConstr(z_d[i][c] <= z_ihc[i][h][c] + (1 - z_h[h][c]), name)
                    print(name)
        print()

        # If (h, c) then (i, c) can only not be selected if some (i, h, c) is not selected
        # for c in range(k):
        #     for h in range(n_h):
        #         for i in range(n_d):
        #             name = "z_D({i}, {c}) <= z_IHC({i}, {h}, {c}) + (1 - z_H({h}, {c}))".format(c=c, h=h, i=i)
        #             m.addConstr(z_d[i][c] <= z_ihc[i][h][c] + (1 - z_h[h][c]), name)
        #             print(name)
        # print()

        # If (i, c) is not selected, then there must exist an h such that: (h, c) and not (i, h, c)
        for c in range(k):
            for i in range(n_d):
                name = "..."  # TODO name
                m.addConstr(sum(z_h[h][c] for h in range(n_h)) - sum(z_ihc[i][h][c] for h in range(n_h)) >= 1 - z_d[i][c], name)
                print(name)
        print()

        # If (i, c) is selected, then there must be at least one selected tuple (h, c)
        for c in range(k):
            for i in range(n_d):
                name = "SUM_(h = 1..n_H) z_H(h, {c}) >= 1 - (1 - z_D({i}, {c}))".format(c=c, i=i)
                m.addConstr(sum(z_h[h][c] for h in range(n_h)) >= 1 - (1 - z_d[i][c]))
                print(name)
        print()

        # If the triple (i, h, c) is selected, then both (i, h) and (h, c) must be true
        for h in range(n_h):
            for c in range(k):
                for i in range(n_d):
                    name = "z_IH({i}, {h}) + z_H({h}, {c}) >= 2 - 2 * (1 - z_IHC({i}, {h}, {c}))".format(h=h, c=c, i=i)
                    m.addConstr(z_ih[i][h] + z_h[h][c] >= 2 - 2 * (1 - z_ihc[i][h][c]), name)
                    print(name)
        print()

        # If the triple (i, h, c) is not selected, then either (i, h) or (h, c) must be false
        for h in range(n_h):
            for c in range(k):
                for i in range(n_d):
                    name = "z_IH({i}, {h}) + z_H({h}, {c}) <= 2 * z_IHC({i}, {h}, {c}) + 1".format(h=h, c=c, i=i)
                    m.addConstr(z_ih[i][h] + z_h[h][c] <= 2 * z_ihc[i][h][c] + 1, name)
                    print(name)
        print()

        # If an example is assigned to any conjunction, it is selected (covered)
        for i in range(n_d):
            for c in range(k):
                name = "s_D({i}) >= z_D({i}, {c})".format(i=i, c=c)
                m.addConstr(s_d[i] >= z_d[i][c], name)
                print(name)
        print()

        # If an example is not assigned to any conjunction, it cannot be selected (covered)
        for i in range(n_d):
            name = "s_D({i}) <= SUM_(c = 1..k) z_D({i}, c)".format(i=i)
            m.addConstr(s_d[i] <= sum(z_d[i][c] for c in range(k)), name)
            print(name)
        print()

        # Examples can only be assigned to active hyperplanes
        # z_ih <= s_h for all i, h
        # for h in range(n_h):
        #     for i in range(n_d):
        #         name = "z_ih({i}, {h}) <= s_h({h})".format(i=i, h=h)
        #         m.addConstr(z_ih[i][h] <= s_h[h], name)
        #         print(name)
        # print()

        # Hack example
        # m.addConstr(a[0][0] == 1)
        # m.addConstr(a[0][1] == 0)
        # m.addConstr(b[0] == 0.5)
        #
        # m.addConstr(a[1][0] == 0)
        # m.addConstr(a[1][1] == 1)
        # m.addConstr(b[1] == 0.5)
        #
        # m.addConstr(a[2][0] == -1)
        # m.addConstr(a[2][1] == 0)
        # m.addConstr(b[2] == -0.5)
        #
        # m.addConstr(a[3][0] == 0)
        # m.addConstr(a[3][1] == -1)
        # m.addConstr(b[3] == -0.5)
        #
        # m.addConstr(s_h[0] == 1)
        # m.addConstr(s_h[1] == 1)
        # m.addConstr(s_h[2] == 1)
        # m.addConstr(s_h[3] == 1)

        # m.addConstr(z_h[0][0] == 1)
        # m.addConstr(z_h[1][0] == 1)
        # m.addConstr(z_h[2][0] == 0)
        # m.addConstr(z_h[3][0] == 0)
        # m.addConstr(z_h[0][1] == 0)
        # m.addConstr(z_h[1][1] == 0)
        # m.addConstr(z_h[2][1] == 1)
        # m.addConstr(z_h[3][1] == 1)

        # m.addConstr(z_ih[0][0] == 1)
        # m.addConstr(z_ih[1][0] == 0)
        # m.addConstr(z_ih[2][0] == 1)
        # m.addConstr(z_ih[3][0] == 0)
        # m.addConstr(z_ih[0][1] == 1)
        # m.addConstr(z_ih[1][1] == 0)
        # m.addConstr(z_ih[2][1] == 0)
        # m.addConstr(z_ih[3][1] == 1)

        # m.addConstr(z_d[0][0] == 1)
        # m.addConstr(z_d[1][0] == 0)
        # m.addConstr(z_d[2][0] == 0)
        # m.addConstr(z_d[3][0] == 0)
        # m.addConstr(z_d[0][1] == 0)
        # m.addConstr(z_d[1][1] == 0)
        # m.addConstr(z_d[2][1] == 1)
        # m.addConstr(z_d[3][1] == 0)

        # Wrong solution
        # m.addConstr(a[0][0] == 1)
        # m.addConstr(a[0][1] == 0)
        # m.addConstr(b[0] == 0.5)
#
        # m.addConstr(a[1][0] == 0)
        # m.addConstr(a[1][1] == 1)
        # m.addConstr(b[1] == 0.5)
#
        # m.addConstr(a[2][0] == -1)
        # m.addConstr(a[2][1] == 0)
        # m.addConstr(b[2] == -0.6)
#
        # m.addConstr(a[3][0] == 0)
        # m.addConstr(a[3][1] == -1)
        # m.addConstr(b[3] == -0.6)
#
        # m.addConstr(s_h[0] == 1)
        # m.addConstr(s_h[1] == 1)
        # m.addConstr(s_h[2] == 1)
        # m.addConstr(s_h[3] == 1)


        m.optimize()

        for v in m.getVars():
            print(v.varName, v.x)

        print('Obj:', m.objVal)

        print()

        # Lets extract the kDNF formulation
        print("Hyperplanes")
        conjunctions = []
        from pysmt.shortcuts import And, Or, Real, LE, Plus, Times, Symbol

        for conj in range(k):
            conjunction = []
            for h in range(n_h):
                if int(z_h[h][conj].x) == 1:
                    coefficients = [Real(float(a[h][j].x)) for j in range(n_f)]
                    linear_sum = Plus([Times(c, domain.get_symbol(v)) for c, v in zip(coefficients, domain.variables)])
                    constant = Real(float(b[h].x))
                    conjunction.append(LE(linear_sum, constant))
            conjunctions.append(conjunction)
        return conjunctions


    @staticmethod
    def _get_misclassification(data):
        true_count = 0
        for _, l in data:
            if l:
                true_count += 1
        return min(true_count, len(data) - true_count)

    @staticmethod
    def _convert(value):
        return float(value.constant_value())
