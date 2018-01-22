from __future__ import print_function

from learner import Learner
import gurobipy.gurobipy as milp


class GreedyRuleLearner(Learner):
    def __init__(self, rule_learner_name, max_hyperplanes_per_rule, max_terms_per_rule):
        Learner.__init__(self, "greedy_{}".format(rule_learner_name))
        self.max_hyperplanes_per_rule = max_hyperplanes_per_rule
        self.max_terms_per_rule = max_terms_per_rule

    def learn(self, domain, data, border_indices):
        pos_uncovered_indices = []
        neg_indices = []
        for i in range(len(data)):
            if data[i][1]:
                pos_uncovered_indices.append(i)
            else:
                neg_indices.append(i)

        dnf_list = []
        iterations = 0
        while len(pos_uncovered_indices) > 0:
            print("{} uncovered positive examples remain".format(len(pos_uncovered_indices)))
            rule_list = self.learn_rule(domain, data, pos_uncovered_indices, neg_indices)
            new_pos_uncovered_indices = []
            for i in pos_uncovered_indices:
                if not Learner.check_example(domain, data[i][0], [rule_list]):
                    new_pos_uncovered_indices.append(i)
            dnf_list.append(rule_list)
            print("Covered {} new positive examples".format(len(pos_uncovered_indices) - len(new_pos_uncovered_indices)))
            pos_uncovered_indices = new_pos_uncovered_indices
            print(rule_list)
            print(pos_uncovered_indices)
            iterations += 1
            if iterations > 0:
                break
        print("All positive examples covered")

        for i in neg_indices:
            if Learner.check_example(domain, data[i][0], dnf_list):
                print("Negative example {}: {} was wrongfully covered".format(i, data[i][0]))
        return dnf_list

    def learn_rule(self, domain, data, pos_uncovered_indices, neg_indices):
        raise NotImplementedError()


class GreedyMilpRuleLearner(GreedyRuleLearner):
    def __init__(self, max_hyperplanes_per_rule, max_terms_per_rule):
        GreedyRuleLearner.__init__(self, "milp", max_hyperplanes_per_rule, max_terms_per_rule)

    def learn_rule(self, domain, data, pos_uncovered_indices, neg_indices):
        # Create a new model
        m = milp.Model("kDNF")

        print("Data")
        for row, l in data:
            print(*(["{}: {:.2f}".format(v, Learner._convert(row[v])) for v in domain.variables] + [l]))
        print()

        # --- Computed constants
        n_r = len(domain.real_vars)
        n_b = len(domain.bool_vars)
        n_h = self.max_hyperplanes_per_rule
        n_e = len(data)
        k = self.max_terms_per_rule
        mu = 0.005

        x_er = [[Learner._convert(row[v]) for v in domain.real_vars] for row, _ in data]
        x_eb = [[Learner._convert(row[v]) for v in domain.bool_vars] for row, _ in data]

        # --- Variables

        # Inequality h is selected for the rule
        s_h = [m.addVar(vtype=milp.GRB.BINARY, name="s_h({h})".format(h=h)) for h in range(n_h)]

        # Example e covered by inequality h
        c_eh = [
            [m.addVar(vtype=milp.GRB.BINARY, name="c_eh({e}, {h})".format(e=e, h=h)) for h in range(n_h)]
            for e in range(n_e)
        ]

        # Example e covered by rule
        c_e = [m.addVar(vtype=milp.GRB.BINARY, name="c_e({e})".format(e=e)) for e in range(n_e)]

        # Boolean feature (or negation) selected
        s_b = [m.addVar(vtype=milp.GRB.BINARY, name="s_b({b})".format(b=b)) for b in range(2 * n_b)]

        # Coefficient of the r-th variable of inequality h
        a_hr = [
            [m.addVar(vtype=milp.GRB.CONTINUOUS, lb=-1, ub=1, name="a_hr({h}, {r})".format(h=h, r=r))
             for r in range(n_r)]
            for h in range(n_h)
        ]

        # Offset of inequality h
        b_h = [m.addVar(vtype=milp.GRB.CONTINUOUS, lb=-1, ub=1, name="b_h({h})".format(h=h)) for h in range(n_h)]
        b_abs_h = [m.addVar(vtype=milp.GRB.CONTINUOUS, lb=-1, ub=1, name="b_abs_h({h})".format(h=h)) for h in range(n_h)]

        # Auxiliary variable (c_eh AND s_h)
        and_eh = [
            [m.addVar(vtype=milp.GRB.BINARY, name="and_eh({e}, {h})".format(e=e, h=h)) for h in range(n_h)]
            for e in range(n_e)
        ]

        # Auxiliary variable (x_eb AND s_b)
        and_eb = [
            [m.addVar(vtype=milp.GRB.BINARY, name="and_eb({e}, {b})".format(e=e, b=b)) for b in range(n_b)]
            for e in range(n_e)
        ]

        print()

        # Set objective
        m.setObjective(sum(c_e[e] for e in range(n_e) if data[e][1]), milp.GRB.MAXIMIZE)

        # Add constraint: and_eh <= c_eh
        for e in range(n_e):
            if not data[e][1]:
                for h in range(n_h):
                    name = "and_eh({e}, {h}) <= c_eh({e}, {h})".format(e=e, h=h)
                    m.addConstr(and_eh[e][h] <= c_eh[e][h], name)
                    print(name)
        print()

        # Add constraint: and_eh <= s_h
        for e in range(n_e):
            if not data[e][1]:
                for h in range(n_h):
                    name = "and_eh({e}, {h}) <= s_h({h})".format(e=e, h=h)
                    m.addConstr(and_eh[e][h] <= s_h[h], name)
                    print(name)
        print()

        # Add constraint: and_eh >= c_eh + s_h - 1
        for e in range(n_e):
            if not data[e][1]:
                for h in range(n_h):
                    name = "and_eh({e}, {h}) >= c_eh({e}, {h}) + s_h({h}) - 1".format(e=e, h=h)
                    m.addConstr(and_eh[e][h] >= c_eh[e][h] + s_h[h] - 1, name)
                    print(name)
        print()

        # Add constraint: and_eb <= x_eb
        for e in range(n_e):
            if not data[e][1]:
                for b in range(2 * n_b):
                    name = "and_eb({e}, {b}) <= x_eb({e}, {b})".format(e=e, b=b)
                    m.addConstr(and_eb[e][b] <= (x_eb[e][b] if b < n_b else 1 - x_eb[e][b]), name)
                    print(name)
        print()

        # Add constraint: and_eb <= s_h
        for e in range(n_e):
            if not data[e][1]:
                for b in range(2 * n_b):
                    name = "and_eb({e}, {b}) <= s_b({b})".format(e=e, b=b)
                    m.addConstr(and_eb[e][b] <= s_b[b], name)
                    print(name)
        print()

        # Add constraint: and_eb >= x_eb + s_b - 1
        for e in range(n_e):
            if not data[e][1]:
                for b in range(2 * n_b):
                    name = "and_eb({e}, {b}) >= x_eb({e}, {b}) + s_b({b}) - 1".format(e=e, b=b)
                    m.addConstr(and_eb[e][b] >= (x_eb[e][b] if b < n_b else 1 - x_eb[e][b]) + s_b[b] - 1, name)
                    print(name)
        print()

        # Add constraint: SUM_(h = 1..n_h) and_eh - s_h + SUM_(b = 1..2 * n_b) and_eb - s_b <= -1
        for e in range(n_e):
            if not data[e][1]:
                name = "SUM_(h = 1..n_h) and_eh({e}, h) - s_h(h) + SUM_(b = 1..2*n_b) and_eb({e}, b) - s_b(b) <= -1"\
                    .format(e=e)
                m.addConstr(sum(and_eh[e][h] - s_h[h] for h in range(n_h))
                            + sum(and_eb[e][b] - s_b[b] for b in range(2 * n_b)) <= -1, name)
                print(name)

        # Add constraint: c_e <= c_eh + (1 - s_h)
        for e in range(n_e):
            if data[e][1]:
                for h in range(n_h):
                    name = "c_e({e}) <= c_eh({e}, {h}) + (1 - s_h({h}))".format(e=e, h=h)
                    m.addConstr(c_e[e] <= c_eh[e][h] + (1 - s_h[h]), name)
                    print(name)

        # Add constraint: c_e <= x_eb + (1 - s_b)
        for e in range(n_e):
            if data[e][1]:
                for b in range(2 * n_b):
                    name = "c_e({e}) <= x_eb({e}, {b}) + (1 - s_b({b}))".format(e=e, b=b)
                    m.addConstr(c_e[e] <= x_eb[e][b] + (1 - s_b[b]), name)
                    print(name)

        # Add constraint: SUM_(r = 1..n_r) a_hr * x_er <= b_h + 2 * (1 - c_eh)
        for e in range(n_e):
            for h in range(n_h):
                name = "SUM_(r = 1..n_r) a_hr({h}, r) * x_er({e}, r) <= b_h({h}) + 2 * (1 - c_eh({e}, {h}))"\
                    .format(e=e, h=h)
                m.addConstr(sum(a_hr[h][r] * x_er[e][r] for r in range(n_r)) <= b_h[h] + 2 * (1 - c_eh[e][h]), name)
                print(name)

        # Add constraint: SUM_(r = 1..n_r) a_hr * x_er >= b_h - mu - 2 * c_eh
        for e in range(n_e):
            for h in range(n_h):
                name = "SUM_(r = 1..n_r) a_hr({h}, r) * x_er({e}, r) >= b_h({h}) - mu - 2 * c_eh({e}, {h})"\
                    .format(e=e, h=h)
                m.addConstr(sum(a_hr[h][r] * x_er[e][r] for r in range(n_r)) >= b_h[h] + mu - 2 * c_eh[e][h], name)
                print(name)

        # Add constraint: SUM_(h = 1..n_h) s_h + SUM_(b = 1..n_b) s_b <= k
        name = "SUM_(h = 1..n_h) s_h(h) + SUM_(b = 1..n_b) s_b(b) <= k"
        m.addConstr(sum(s_h[h] for h in range(n_h)) + sum(s_b[b] for b in range(2 * n_b)) <= k, name)
        print(name)

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
        # m.addConstr(z_ih[0][2] == 0)
        # m.addConstr(z_ih[1][2] == 1)
        # m.addConstr(z_ih[2][2] == 0)
        # m.addConstr(z_ih[3][2] == 1)
        # m.addConstr(z_ih[0][3] == 0)
        # m.addConstr(z_ih[1][3] == 1)
        # m.addConstr(z_ih[2][3] == 1)
        # m.addConstr(z_ih[3][3] == 0)

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

        # m.addConstr(s_h[0] == 1)
        # m.addConstr(s_h[1] == 1)
        # m.addConstr(s_h[2] == 0)
        # m.addConstr(s_h[3] == 0)
#
        # m.addConstr(a_hr[0][0] == -1)
        # m.addConstr(a_hr[0][1] == -0.5)
        # m.addConstr(b_h[0] == -0.2)
#
        # m.addConstr(a_hr[1][0] == 1)
        # m.addConstr(a_hr[1][1] == 0)
        # m.addConstr(b_h[1] == 0.75)

        m.optimize()

        for v in m.getVars():
            print(v.varName, v.x)

        print('Obj:', m.objVal)

        print()

        # Lets extract the rule formulation
        rule = []
        from pysmt.shortcuts import Real, LE, Plus, Times, Not

        for h in range(n_h):
            if int(s_h[h].x) == 1:
                coefficients = [Real(float(a_hr[h][r].x)) for r in range(n_r)]
                constant = Real(float(b_h[h].x))
                linear_sum = Plus([Times(c, domain.get_symbol(v)) for c, v in zip(coefficients, domain.real_vars)])
                rule.append(LE(linear_sum, constant))

        for b in range(2 * n_b):
            if int(s_b[b].x) == 1:
                var = domain.get_symbol(domain.bool_vars[b if b < n_b else b - n_b])
                rule.append(var if b < n_b else Not(var))

        print("Features", x_er[4][0], x_er[4][1])
        print(sum(a_hr[1][r].x * x_er[4][r] for r in range(n_r)), "<=", b_h[1].x + 2 * (1 - c_eh[4][1].x))
        print(sum(a_hr[1][r].x * x_er[4][r] for r in range(n_r)), ">=", b_h[1].x + mu - 2 * c_eh[4][1].x)

        return rule

