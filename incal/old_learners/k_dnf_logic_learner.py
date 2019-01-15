from __future__ import print_function

import itertools

import numpy as np
from pysmt.shortcuts import Plus, Real, Times, LE, GE
from pysmt.typing import REAL, BOOL

from learner import Learner


class KDNFLogicLearner(Learner):
    def __init__(self, k):
        Learner.__init__(self)
        self.k = k

    def learn(self, domain, data, border_indices):
        positive_indices = [i for i in range(len(data)) if data[i][1]]
        real_vars = [v for v in domain.variables if domain.var_types[v] == REAL]
        bool_vars = [v for v in domain.variables if domain.var_types[v] == BOOL]
        d = len(real_vars)
        hyperplanes = []
        for indices in itertools.combinations(positive_indices, d):
            print(indices)
            hyperplanes.append(Learner.fit_hyperplane(domain, [data[i][0] for i in indices]))
        boolean_data = []
        for i in range(len(data)):
            row = []
            for v in bool_vars:
                row.append(data[i][0][v].constant_value())
            boolean_data.append(row)
        hyperplanes_smt = []
        for a, c in hyperplanes:
            lhs_smt = Plus(Times(Real(float(a[j])), domain.get_symbol(real_vars[j])) for j in range(d))
            hyperplanes_smt.append(LE(lhs_smt, Real(c)))
            lhs_smt = Plus(Times(Real(-float(a[j])), domain.get_symbol(real_vars[j])) for j in range(d))
            hyperplanes_smt.append(LE(lhs_smt, Real(-c)))
            for i in range(len(data)):
                lhs = 0
                for j in range(d):
                    lhs += float(a[j]) * float(data[i][0][real_vars[j]].constant_value())
                boolean_data[i].append(lhs <= c)
                boolean_data[i].append(lhs >= c)
        print(boolean_data)
        # logical_dnf_indices = [[i] for i in range(len(boolean_data[0]))]
        logical_dnf_indices = self.learn_logical(boolean_data, [row[1] for row in data])
        logical_dnf = [
            [domain.get_symbol(bool_vars[i]) if i < len(bool_vars) else
             hyperplanes_smt[i - len(bool_vars)] for i in conj_indices]
            for conj_indices in logical_dnf_indices
        ]
        print(logical_dnf)
        return logical_dnf

    def learn_logical(self, boolean_data, labels):
        conjunctions = []
        for k in range(1, self.k + 1):
            for features in itertools.combinations(list(range(len(boolean_data))), k):
                accept = True
                for entry, label in zip(boolean_data, labels):
                    if not label and all(entry[j] for j in features):
                        accept = False
                        break
                if accept:
                    conjunctions.append(features)
        return conjunctions


class GreedyMaxRuleLearner(KDNFLogicLearner):
    def __init__(self, max_literals):
        KDNFLogicLearner.__init__(self, max_literals)

    def learn_logical(self, boolean_data, labels):
        attributes = np.matrix(boolean_data)
        examples = attributes.shape[0]
        features = attributes.shape[1]
        conjunctions = []
        counts = np.sum(attributes, axis=0).A1
        print(examples, features, counts.shape)

        return []


class GreedyLogicDNFLearner(KDNFLogicLearner):
    def __init__(self, max_terms, max_literals):
        KDNFLogicLearner.__init__(self, max_literals)
        self.max_terms = max_terms

    @property
    def max_literals(self):
        return self.k

    def learn_logical(self, boolean_data, labels):
        attributes = np.matrix(boolean_data)
        examples = attributes.shape[0]
        features = attributes.shape[1]
        conjunctions = []
        counts = np.sum(attributes, axis=0).A1
        print(counts[0])

        for i in range(self.max_terms):
            lb = 0
            ub = examples
            candidates = [([], examples)]
            new_candidates = []
            while len(candidates) > 0:
                for pattern, count in candidates:
                    start_index = 0 if len(pattern) == 0 else max(pattern) + 1
                    covered = [i for i in range(examples) if all(attributes[i, j] for j in pattern)]
                    pos_covered = [i for i in covered if labels[i]]
                    neg_covered = [i for i in covered if not labels[i]]

                    for j in range(start_index, self.max_literals):
                        if counts[j] > lb:
                            pass
                for j in range(self.max_literals):
                    for features in itertools.combinations(list(range(len(boolean_data))), self.k):
                        accept = True
                        for entry, label in zip(boolean_data, labels):
                            if not label and all(entry[j] for j in features):
                                accept = False
                                break
                        if accept:
                            conjunctions.append(features)
        return conjunctions
