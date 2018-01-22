import matplotlib as mpl
import os

from incremental_learner import IncrementalObserver
from smt_check import SmtChecker

mpl.use('TkAgg')
import matplotlib.pyplot as plt


class PlottingObserver(IncrementalObserver):
    def __init__(self, data, directory, name, feat_x, feat_y):
        self.data = data
        self.directory = directory

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.name = name
        self.all_active = set()
        self.feat_x = feat_x
        self.feat_y = feat_y
        self.iteration = 0

    def observe_initial(self, initial_indices):
        self.all_active = self.all_active.union(initial_indices)
        name = "{}{}{}_{}".format(self.directory, os.path.sep, self.name, self.iteration)
        draw_border_points(self.feat_x, self.feat_y, self.data, initial_indices, name)

    def observe_iteration(self, theory, new_active_indices, solving_time, selection_time):
        self.iteration += 1
        learned_labels = [SmtChecker(instance).check(theory) for instance, _ in self.data]
        name = "{}{}{}_{}".format(self.directory, os.path.sep, self.name, self.iteration)
        draw_points(self.feat_x, self.feat_y, self.data, learned_labels, name, self.all_active, new_active_indices)
        self.all_active = self.all_active.union(new_active_indices)


def draw_border_points(feat_x, feat_y, data, border_indices, name):
    relevant_pos = []
    irrelevant_pos = []
    relevant_neg = []
    irrelevant_neg = []

    for i in range(len(data)):
        row = data[i]
        point = (row[0][feat_x], row[0][feat_y])
        if i in border_indices and row[1]:
            relevant_pos.append(point)
        elif i in border_indices and not row[1]:
            relevant_neg.append(point)
        elif i not in border_indices and row[1]:
            irrelevant_pos.append(point)
        else:
            irrelevant_neg.append(point)

    if len(relevant_pos) > 0: plt.scatter(*zip(*relevant_pos), c="blue", marker="o")
    if len(irrelevant_pos) > 0: plt.scatter(*zip(*irrelevant_pos), c="blue", marker="o", alpha=0.2)
    if len(relevant_neg) > 0: plt.scatter(*zip(*relevant_neg), c="grey", marker="x")
    if len(irrelevant_neg) > 0: plt.scatter(*zip(*irrelevant_neg), c="grey", marker="x", alpha=0.2)

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.savefig("{}.png".format(name))
    plt.gcf().clear()


def draw_points(feat_x, feat_y, data, learned_labels, name, active_indices, new_active_indices, hyperplanes=None):
    points = []

    for i in range(len(data)):
        instance, label = data[i]
        point = (float(instance[feat_x]), float(instance[feat_y]))
        correct = learned_labels[i] == label
        status = "active" if i in active_indices else ("new_active" if i in new_active_indices else "excluded")
        points.append((label, correct, status, point))

    label_markers = {True: "o", False: "x"}
    correctness_colors = {True: "green", False: "red"}
    for label in [True, False]:
        for correct in [True, False]:
            for status, alpha in [("active", 1), ("new_active", 1), ("excluded", 0.2)]:
                marker = label_markers[label]
                color = correctness_colors[correct]
                if status == "active":
                    color = "black"
                selection = [p for l, c, s, p in points if l == label and c == correct and s == status]
                if len(selection) > 0:
                    plt.scatter(*zip(*selection), c=color, marker=marker, alpha=alpha)

    if hyperplanes is not None:
        planes = [constraint_to_hyperplane(h) for conj in hyperplanes for h in conj if h.is_le() or h.is_lt()]
        for plane in planes:
            print(plane)
            if plane[0][feat_y] == 0:
                plt.plot([plane[1], plane[1]], [0, 1])
            else:
                plt.plot([0, 1], [(plane[1] - plane[0][feat_x] * x) / plane[0][feat_y] for x in [0, 1]])

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.savefig("{}.png".format(name))
    plt.gcf().clear()


def constraint_to_hyperplane(constraint):
    if constraint.is_le() or constraint.is_lt():
        coefficients = dict()
        left, right = constraint.args()
        if right.is_plus():
            left, right = right, left
        if left.is_plus():
            for term in left.args():
                if term.is_times():
                    c, v = term.args()
                    coefficients[v.symbol_name()] = float(c.constant_value())
                else:
                    raise RuntimeError("Unexpected value, expected product, was {}".format(term))
        else:
            raise RuntimeError("Unexpected value, expected sum, was {}".format(left))
        return coefficients, float(right.constant_value())
    raise RuntimeError("Unexpected constraint, expected inequality, was {}".format(constraint))
