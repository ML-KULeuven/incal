import itertools
import time

import os
import pysmt.shortcuts as smt

from incremental_learner import IncrementalObserver
from visualize import RegionBuilder

from pywmi.smt_check import evaluate_assignment
from pywmi.plot import plot_data, plot_combined


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
        plot_combined(self.feat_x, self.feat_y, self.domain, None, self.data, None, name, [], initial_indices,
                      condition=self.condition)

    def observe_iteration(self, theory, new_active_indices, solving_time, selection_time):
        self.iteration += 1
        learned_labels = [evaluate_assignment(theory, instance) for instance, _ in self.data]
        name = "{}{}{}_{}".format(self.directory, os.path.sep, self.name, self.iteration)
        plot_combined(self.feat_x, self.feat_y, self.domain, theory, self.data, learned_labels, name, self.all_active,
                      new_active_indices, condition=self.condition)
        self.all_active = self.all_active.union(new_active_indices)
