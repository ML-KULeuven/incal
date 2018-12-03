import itertools
import math
import time

import matplotlib as mpl
import os
import pysmt.shortcuts as smt

from incremental_learner import IncrementalObserver
from smt_check import SmtChecker
from smt_print import pretty_print
from visualize import RegionBuilder

mpl.use('TkAgg')
import matplotlib.pyplot as plt


class PlottingObserver(IncrementalObserver):
    def __init__(self, domain, data, directory, name, feat_x, feat_y, condition=None, auto_clean=False, run_name=None):
        self.domain = domain
        self.data = data

        if not os.path.exists(directory):
            os.makedirs(directory)

        if auto_clean:
            run_number = 0
            run_dir = None
            while run_dir is None or os.path.exists(run_dir):
                date_folders = time.strftime("%Y{s}%m{s}%d{s}".format(s=os.path.sep))
                run_name = run_name + " " if run_name is not None else ""
                run_dir_name = "run {}{}".format(run_name, time.strftime("%Hh %Mm %Ss"))
                run_dir = os.path.join(directory, date_folders, run_dir_name)
                if run_number > 0:
                    run_dir += "_{}".format(run_number)
                run_number += 1
            os.makedirs(run_dir)
            directory = run_dir

        self.directory = directory

        self.name = name
        self.all_active = set()
        self.feat_x = feat_x
        self.feat_y = feat_y
        self.iteration = 0
        self.condition = condition

    def observe_initial(self, initial_indices):
        self.all_active = self.all_active.union(initial_indices)
        name = "{}{}{}_{}".format(self.directory, os.path.sep, self.name, self.iteration)
        draw_points(self.feat_x, self.feat_y, self.domain, None, self.data, None, name, [], initial_indices,
                    condition=self.condition)

    def observe_iteration(self, theory, new_active_indices, solving_time, selection_time):
        self.iteration += 1
        learned_labels = [SmtChecker(instance).check(theory) for instance, _ in self.data]
        name = "{}{}{}_{}".format(self.directory, os.path.sep, self.name, self.iteration)
        draw_points(self.feat_x, self.feat_y, self.domain, theory, self.data, learned_labels, name, self.all_active,
                    new_active_indices, condition=self.condition)
        self.all_active = self.all_active.union(new_active_indices)


def draw_points(feat_x, feat_y, domain, formula, data, learned_labels, name, active_indices, new_active_indices,
                condition=None):

    row_vars = domain.bool_vars[:int(len(domain.bool_vars) / 2)]
    col_vars = domain.bool_vars[int(len(domain.bool_vars) / 2):]
    sf_size = 2

    fig = plt.figure(num=None, figsize=(2 ** len(col_vars) * sf_size, 2 ** len(row_vars) * sf_size), dpi=300)

    for assignment in itertools.product([True, False], repeat=len(domain.bool_vars)):
        row = 0
        for b in assignment[:len(row_vars)]:
            row = row * 2 + (1 if b else 0)

        col = 0
        for b in assignment[len(row_vars):]:
            col = col * 2 + (1 if b else 0)

        index = row * (2 ** len(col_vars)) + col
        ax = fig.add_subplot(2 ** len(row_vars), 2 ** len(col_vars), 1 + index)

        if formula is not None:
            substitution = {domain.get_symbol(v): smt.Bool(a) for v, a in zip(domain.bool_vars, assignment)}
            substituted = formula.substitute(substitution)
            region = RegionBuilder(domain).walk_smt(substituted)
            try:
                if region.dim == 2:
                    region.plot(ax=ax, color="green", alpha=0.2)
            except IndexError:
                pass

        points = []
        for i in range(len(data)):
            instance, label = data[i]
            point = (float(instance[feat_x]), float(instance[feat_y]))
            correct = (learned_labels[i] == label) if learned_labels is not None else True
            status = "active" if i in active_indices else ("new_active" if i in new_active_indices else "excluded")
            match = all(instance[v] == a for v, a in zip(domain.bool_vars, assignment))
            if match and (condition is None or condition(instance, label)):
                points.append((label, correct, status, point))

        def get_color(_l, _c, _s):
            if _s == "active":
                return "black"
            return "green" if _l else "red"

        def get_marker(_l, _c, _s):
            # if _s == "active":
            #     return "v"
            return "+" if _l else "."

        def get_alpha(_l, _c, _s):
            if _s == "active":
                return 0.5
            elif _s == "new_active":
                return 1
            elif _s == "excluded":
                return 0.2

        for label in [True, False]:
            for correct in [True, False]:
                for status in ["active", "new_active", "excluded"]:
                    marker, color, alpha = [f(label, correct, status) for f in (get_marker, get_color, get_alpha)]
                    selection = [p for l, c, s, p in points if l == label and c == correct and s == status]
                    if len(selection) > 0:
                        ax.scatter(*zip(*selection), c=color, marker=marker, alpha=alpha)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.savefig("{}.png".format(name))
    plt.close(fig)
