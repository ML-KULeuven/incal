from __future__ import print_function, division

import json

from pywmi import smt_to_nested
from pywmi.smt_print import pretty_print

from incal.incremental_learner import IncrementalObserver


class LoggingObserver(IncrementalObserver):
    def __init__(self, filename, k, h, seed=None, verbose=True, violation_counter=None):
        self.filename = filename
        self.verbose = verbose
        self.violation_counter = violation_counter
        self.k = k
        self.h = h

        if filename is not None:
            with open(self.filename, "w") as f:
                print("", file=f, end="")

        if seed is not None:
            self.log({"type": "seed", "seed": seed, "k": self.k, "h": self.h})

    def log(self, flat):
        if self.filename is not None:
            with open(self.filename, "a") as f:
                print(json.dumps(flat), file=f)

    def observe_initial(self, initial_indices):
        flat = {"type": "initial", "indices": initial_indices, "k": self.k, "h": self.h}
        if self.verbose:
            print("Starting with {} examples".format(len(initial_indices)))
        self.log(flat)

    def observe_iteration(self, theory, new_active_indices, solving_time, selection_time):
        flat = {
            "type": "update",
            "theory": smt_to_nested(theory),
            "indices": [int(v) for v in new_active_indices],
            "solving_time": solving_time,
            "selection_time": selection_time,
            "k": self.k,
            "h": self.h,
        }
        if self.violation_counter is not None:
            flat["violations"] = [int(v) for v in self.violation_counter.last_violations]

        if self.verbose:
            print("Found model after {:.2f}s".format(solving_time))
            print(pretty_print(theory))
            if self.violation_counter is not None:
                violation_count = len(self.violation_counter.last_violations)
                selected_count = len(new_active_indices)
                print("Selected {} of {} violations in {:.2f}s".format(selected_count, violation_count, selection_time))
        self.log(flat)
