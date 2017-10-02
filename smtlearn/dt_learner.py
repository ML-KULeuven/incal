from __future__ import print_function

from gurobipy.gurobipy import *


# noinspection PyPep8Naming
class OCTLearner(object):
    def __init__(self, N_min, D, alpha):
        self.N_min = N_min
        self.D = D
        self.alpha = alpha
        self.mu = 0.005

    def learn(self, domain, data):
        # Create a new model
        m = Model("OCT")

        print("Data")
        for row, l in data:
            print(*(["{}: {:.2f}".format(v, OCTLearner._convert(row[v])) for v in domain.variables] + [l]))
        print()

        # Computed constants
        L_hat = OCTLearner._get_misclassification(data)

        print("Majority label misclassification cost", L_hat)

        p = len(domain.variables)
        K = 2
        n = len(data)
        leaf_count = 2 ** self.D
        inline_count = leaf_count - 1

        T = list(range(leaf_count + inline_count))
        # T_B = [t for t in T if t < inline_count]
        # T_L = [t for t in T if t >= inline_count]

        print("Nodes: {} internal, {} leaves".format(inline_count, leaf_count))

        Y_ik = [
            [1 if data[i][1] else -1, -1 if data[i][1] else 1]
            for i in range(n)
        ]

        A_L = [[] for _ in range(inline_count + leaf_count)]  # TODO
        A_R = [[] for _ in range(inline_count + leaf_count)]  # TODO

        def populate_ancestors(index):
            left = (index + 1) * 2 - 1
            right = (index + 1) * 2
            if right < inline_count + leaf_count:
                A_L[left] = A_L[index] + [index]
                A_R[left] = A_R[index]
                A_L[right] = A_L[index]
                A_R[right] = A_R[index] + [index]
                populate_ancestors(left)
                populate_ancestors(right)
        populate_ancestors(0)

        print("A_L: {}, A_R: {}".format(A_L, A_R))

        x_ij = [[OCTLearner._convert(row[v]) for v in domain.variables] for row, _ in data]
        p_t = [None if i == 0 else int((i + 1) / 2) - 1 for i in range(inline_count)]

        # Variables
        # x = m.addVar(vtype=GRB.BINARY, name="x")
        L_t = [m.addVar(vtype=GRB.INTEGER, name="L_{}".format(i)) for i in range(leaf_count)]
        s_jt = [
            [m.addVar(vtype=GRB.BINARY, name="s{}_{}".format(j, i)) for i in range(inline_count)]
            for j in range(p)
        ]
        N_t = [m.addVar(vtype=GRB.INTEGER, name="N_{}".format(i)) for i in range(leaf_count)]
        N_kt = [
            [m.addVar(vtype=GRB.INTEGER, name="N_{}_{}".format(k, i)) for i in range(leaf_count)]
            for k in range(K)
        ]
        c_kt = [
            [m.addVar(vtype=GRB.BINARY, name="c_{}_{}".format(k, i)) for i in range(leaf_count)]
            for k in range(K)
        ]
        z_it = [
            [m.addVar(vtype=GRB.BINARY, name="z_{}_{}".format(i, t)) for t in range(leaf_count)]
            for i in range(n)
        ]
        l_t = [m.addVar(vtype=GRB.BINARY, name="l_{}".format(t)) for t in range(leaf_count)]
        a_jt = [
            [m.addVar(vtype=GRB.CONTINUOUS, lb=-1, ub=1, name="a_{}_{}".format(j, t)) for t in range(inline_count)]
            for j in range(p)
        ]
        a_hat_jt = [
            [m.addVar(vtype=GRB.CONTINUOUS, name="a_hat_{}_{}".format(j, t)) for t in range(inline_count)]
            for j in range(p)
        ]
        b_t = [m.addVar(vtype=GRB.CONTINUOUS, name="b_{}".format(t)) for t in range(inline_count)]
        d_t = [m.addVar(vtype=GRB.BINARY, name="d_{}".format(t)) for t in range(inline_count)]
        print()

        # Set objective
        m.setObjective(1.0 / L_hat * sum(L_t) + self.alpha * sum(sum(entry) for entry in s_jt), GRB.MINIMIZE)

        # Add constraint: L_t >= N_t - N_kt - n(1 - c_kt) forall k = 1..K, t in T_L
        for k in range(K):
            for t in range(leaf_count):
                name = "L_{t} >= N_{t} - N_{k}_{t} - n(1 - c_{k}_{t})".format(k=k, t=t)
                m.addConstr(L_t[t] >= N_t[t] - N_kt[k][t] - n * (1 - c_kt[k][t]), name)
                print(name)
        print()

        # Add constraint: L_t <= N_t - N_kt + n * c_kt forall k = 1..K, t in T_L
        for k in range(K):
            for t in range(leaf_count):
                name = "L_{t} <= N_{t} - N_{k}_{t} + n * c_{k}_{t}".format(k=k, t=t)
                m.addConstr(L_t[t] <= N_t[t] - N_kt[k][t] + n * c_kt[k][t], name)
                print(name)
        print()

        # Add constraint: L_t >= 0 forall t in T_l
        for t in range(leaf_count):
            name = "L_{t} >= 0".format(t=t)
            m.addConstr(L_t[t] >= 0, name)
            print(name)
        print()

        # Add constraint: N_kt = 1/2 * Sum_i = 1..n (1 + Y_ik) * z_it forall k = 1..K, t in T_L
        for k in range(K):
            for t in range(leaf_count):
                name = "N_{k}_{t} = 1/2 * SUM_(i = 0..{n}) (1 + Y_i_{k}) * z_i_{t}".format(k=k, t=t, n=n-1)
                m.addConstr(N_kt[k][t] == sum((1 + Y_ik[i][k]) * z_it[i][t] for i in range(n)) / 2, name)
                print(name)
        print()

        # Add constraint: N_t = Sum_i = 1..n z_it forall t in T_L
        for t in range(leaf_count):
            name = "N_{t} = SUM_(i = 0..{n}) z_i_{t}".format(t=t, n=n-1)
            m.addConstr(N_t[t] == sum(z_it[i][t] for i in range(n)), name)
            print(name)
        print()

        # Add constraint: Sum_k = 1..K c_kt = l_t forall t in T_L
        for t in range(leaf_count):
            # print(sum([c_kt[k][t] for k in range(K)]))
            name = "SUM_(k = 0..{K}) c_k_{t} = l_{t}".format(t=t, K=K-1)
            m.addConstr(sum(c_kt[k][t] for k in range(K)) == l_t[t], name)
            print(name)
        print()

        # Add constraint: Sum_j = 1..p (a_jm^T * x_ji) + mu <= b_m + (2 + mu) * (1 - z_it)
        # forall i in 1..n, t in T_B, m in A_L(t)
        for i in range(n):
            for t in range(leaf_count):
                for _m in A_L[inline_count + t]:
                    vector_product = sum(a_jt[j][_m] * x_ij[i][j] for j in range(p))
                    name = "SUM_(j = 0..{p}) (a_j_{m}^T * x_j_{i}) + mu <= b_{m} + (2 + mu) * (1 - z_{i}_{t})"\
                        .format(i=i, t=t, m=_m, p=p-1)
                    m.addConstr(vector_product + self.mu <= b_t[_m] + (2 + self.mu) * (1 - z_it[i][t]), name)
                    print(name)
        print()

        # Add constraint: Sum_j = 1..p a_jm^T * x_ji >= b_m - 2 * (1 - z_it) forall i in 1..n, t in T_B, m in A_R(t)
        for i in range(n):
            for t in range(leaf_count):
                for _m in A_R[inline_count + t]:
                    name = "SUM_(j = 0..{p}) a_j_{m}^T * x_j_{i} >= b_{m} - 2 * (1 - z_{i}_{t})"\
                        .format(i=i, t=t, m=_m, p=p-1)
                    vector_product = sum(a_jt[j][_m] * x_ij[i][j] for j in range(p))
                    m.addConstr(vector_product >= b_t[_m] - 2 * (1 - z_it[i][t]), name)
                    print(name)
        print()

        # Add constraint: Sum_t in T_L z_it = 1 forall i = 1..n
        for i in range(n):
            name = "SUM_(t in {T_L}) z_{i}_t = 1".format(i=i, T_L=list(range(leaf_count)))
            m.addConstr(sum(z_it[i][t] for t in range(leaf_count)) == 1, name)
            print(name)
        print()

        # Add constraint: z_it <= l_t forall i = 1..n, t in T_L
        for i in range(n):
            for t in range(leaf_count):
                name = "z_{i}_{t} <= l_{t}".format(i=i, t=t)
                m.addConstr(z_it[i][t] <= l_t[t], name)
                print(name)
        print()

        # Add constraint: Sum_i = 1..n z_it >= N_min * l_t forall t in T_L
        for t in range(leaf_count):
            name = "SUM_(i = 0..{n}) z_i_{t} >= N_min * l_{t}".format(t=t, n=n-1)
            m.addConstr(sum(z_it[i][t] for i in range(n)) >= self.N_min * l_t[t], name)
            print(name)
        print()

        # Add constraint: Sum_j = 1..p a_hat_jt <= d_t forall t in T_B
        for t in range(inline_count):
            name = "SUM_(j = 0..{p}) a_hat_j_{t} <= d_{t}".format(t=t, p=p-1)
            m.addConstr(sum(a_hat_jt[j][t] for j in range(p)) <= d_t[t], name)
            print(name)
        print()

        # Add constraint: a_hat_jt >= a_jt forall j = 1..p, t in T_B
        for j in range(p):
            for t in range(inline_count):
                name = "a_hat_{j}_{t} >= a_{j}_{t}".format(j=j, t=t)
                m.addConstr(a_hat_jt[j][t] >= a_jt[j][t], name)
                print(name)
            print()

        # Add constraint: a_hat_jt >= -a_jt forall j = 1..p, t in T_B
        for j in range(p):
            for t in range(inline_count):
                name = "a_hat_{j}_{t} >= -a_{j}_{t}".format(j=j, t=t)
                m.addConstr(a_hat_jt[j][t] >= -a_jt[j][t], name)
                print(name)
            print()

        # Add constraint: -s_jt <= a_jt <= s_jt forall j = 1..p, t in T_B
        for j in range(p):
            for t in range(inline_count):
                name1 = "-s_{j}_{t} <= a_{j}_{t}".format(j=j, t=t)
                m.addConstr(-s_jt[j][t] <= a_jt[j][t], name1)
                name2 = "s_{j}_{t} >= a_{j}_{t}".format(j=j, t=t)
                m.addConstr(s_jt[j][t] >= a_jt[j][t], name2)
                print(name1)
                print(name2)
        print()

        # Add constraint: s_jt <= d_t forall j = 1..p, t in T_B
        for j in range(p):
            for t in range(inline_count):
                name = "s_{j}_{t} <= d_{t}".format(j=j, t=t)
                m.addConstr(s_jt[j][t] <= d_t[t], name)
                print(name)
        print()

        # Add constraint: Sum_j = 1..p s_jt >= d_t forall t in T_B
        for t in range(inline_count):
            name = "SUM_(j = 0..{p}) s_j_{t} >= d_{t}".format(t=t, p=p-1)
            m.addConstr(sum(s_jt[j][t] for j in range(p)) >= d_t[t], name)
            print(name)
        print()

        # Add constraint: -d_t <= b_t <= d_t forall t in T_B
        for t in range(inline_count):
            name1 = "-d_{t} <= b_{t}".format(t=t)
            m.addConstr(-d_t[t] <= b_t[t], name1)
            name2 = "d_{t} >= b_{t}".format(t=t)
            m.addConstr(d_t[t] >= b_t[t], name2)
            print(name1)
            print(name2)
        print()

        # Add constraint: d_t <= d_p(t) forall t in T_B \ {1}
        for t in range(1, inline_count):
            name = "d_{t} <= d_{p_t}".format(t=t, p_t=p_t[t])
            m.addConstr(d_t[t] <= d_t[p_t[t]], name)
            print(name)
        print()

        # Hack example
        # m.addConstr(N_t[0] == 2)
        # m.addConstr(N_t[1] == 1)

        # m.addConstr(z_it[0][0] == 1)
        # m.addConstr(z_it[1][0] == 1)
        # m.addConstr(z_it[2][0] == 0)
        # m.addConstr(z_it[0][1] == 0)
        # m.addConstr(z_it[1][1] == 0)
        # m.addConstr(z_it[2][1] == 1)

        # m.addConstr(c_kt[0][0] == 1)
        # m.addConstr(c_kt[1][0] == 0)
        # m.addConstr(c_kt[0][1] == 0)
        # m.addConstr(c_kt[1][1] == 1)

        # m.addConstr(a_jt[0][0] == 1)
        # m.addConstr(b_t[0] == 0.6)

        m.optimize()

        for v in m.getVars():
            print(v.varName, v.x)

        print('Obj:', m.objVal)

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
