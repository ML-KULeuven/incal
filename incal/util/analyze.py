import collections
from argparse import ArgumentParser
from typing import List, Optional, Union, Any

import numpy as np

from .options import Experiment
from .plot import ScatterData


def table(data, column_header=None, row_header=None):
    # type: (Union[List[List[Any]], np.ndarray], Optional[List[Any]], Optional[List[Any]]) -> str

    if isinstance(data, np.ndarray):
        data = [[data[row, col] for col in range(data.shape[1])] for row in range(data.shape[0])]

    if row_header is not None:
        if not isinstance(row_header[0], (list, tuple)):
            row_header = [row_header]
        for rh in reversed(row_header):
            row_strings = ["{}".format(rh[i]) for i in range(len(data))]
            max_length = max(len(rs) for rs in row_strings)
            data = [[row_strings[i]] + data[i] for i in range(len(data))]
            if column_header is not None:
                column_header = [" " * (max_length - 1) + " "] + list(column_header)

    if column_header is not None:
        column_header = ["{}".format(e) for e in column_header]
    data = [["{}".format(e) for e in row] for row in data]

    max_lengths = [max(len(data[r][c]) for r in range(len(data))) for c in range(len(data[0]))]
    if column_header is not None:
        max_lengths = [max(max_lengths[c], len(column_header[c])) for c in range(len(data[0]))]

    if column_header is not None:
        data.insert(0, column_header)
        data.insert(1, ["=" * max_lengths[i] for i, c in enumerate(column_header)])

    data = [[row[c] + " " * (max_lengths[c] - len(row[c])) for c in range(len(data[0]))] for row in data]

    return "\n".join("  ".join(e for e in row) for row in data)


def add_arguments(parser: ArgumentParser):
    parser.add_argument("-t", "--targets", nargs="+", type=str, default=None)
    parser.add_argument("-g", "--group_by", nargs="+", type=str, default=None)
    parser.add_argument("-a", "--aggregator", type=str, default=None)
    parser.add_argument("-s", "--sort", type=str, default=None)
    parser.add_argument("-e", "--exclude", nargs="+", type=str, default=None)
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-o", "--options", nargs="+", type=str, default=None)
    parser.add_argument("-w", "--write_to", type=str, default=None)


def mean(iterable):
    return sum(iterable) / len(iterable)


def mean_scores(scores):
    return [mean([scores[r][c] for r in range(len(scores))]) for c in range(len(scores[0]))]


def std_scores(scores):
    return [np.std(np.array([scores[r][c] for r in range(len(scores))])) for c in range(len(scores[0]))]


def parse_args(args):
    return args.targets, args.group_by, args.aggregator, args.sort, args.exclude, args.plot, args.options, args.write_to


def show_from_args(experiments, args):
    show(experiments, *parse_args(args))


def get_property(index, experiment, property_name):
    parts = property_name.split("__")
    if len(parts) > 1:
        operator = parts[-1]
        remaining = "__".join(parts[:-1])
        if operator == "mean":
            return mean(get_property(index, experiment, remaining))
        if operator == "len":
            return len(get_property(index, experiment, remaining))
        if operator == "sum":
            return sum(get_property(index, experiment, remaining))
        if operator == "max":
            return max(get_property(index, experiment, remaining))
        if operator == "min":
            return min(get_property(index, experiment, remaining))
        if operator.startswith("batch"):
            bin_size = float(operator[5:])
            return int(get_property(index, experiment, remaining) / bin_size) * bin_size - bin_size / 2
        else:
            try:
                index = int(operator)
                return get_property(index, experiment, remaining)[index]
            except ValueError:
                pass

    if "." in property_name:
        parts = property_name.split(".")
        return getattr(get_property(index, experiment, ".".join(parts[:-1])), parts[-1])

    if property_name == "id":
        return index

    if property_name == "*":
        return "all"

    if property_name in experiment.parameters.options:
        return experiment.parameters.original_values[property_name]

    if property_name in experiment.results.options:
        return experiment.results.original_values[property_name]

    if experiment.config and property_name in experiment.config.options:
        return experiment.config.original_values[property_name]

    if property_name in experiment.derived:
        return experiment.derived[property_name](experiment)

    try:
        return 1 if is_excluded_from_string(property_name, experiment) else 0
    except ValueError:
        pass

    properties = ["id", "*"] \
                 + sorted(experiment.parameters.options.keys()) \
                 + sorted(experiment.results.options.keys()) \
                 + (sorted(experiment.config.options.keys()) if experiment.config else [])

    raise ValueError("Could not find property: {}, choose a valid property ({}) or a filter"
                     .format(property_name, properties))


def is_excluded_from_string(filter_string, experiment):
    if "<=" in filter_string:
        parts = filter_string.split("<=", 2)
        value = float(parts[1])
        return get_property(0, experiment, parts[0]) <= value
    if ">=" in filter_string:
        parts = filter_string.split(">=", 2)
        value = float(parts[1])
        return get_property(0, experiment, parts[0]) >= value
    if "<" in filter_string:
        parts = filter_string.split("<", 2)
        value = float(parts[1])
        return get_property(0, experiment, parts[0]) < value
    if ">" in filter_string:
        parts = filter_string.split(">", 2)
        value = float(parts[1])
        return get_property(0, experiment, parts[0]) > value
    if "!=" in filter_string:
        parts = filter_string.split("!=", 2)
        value = float(parts[1])
        return get_property(0, experiment, parts[0]) != value
    if "=" in filter_string:
        parts = filter_string.split("=", 2)
        value = float(parts[1])
        return get_property(0, experiment, parts[0]) == value

    raise ValueError("Could not parse filter string {}".format(filter_string))


def is_excluded(experiment: Experiment, exclude: Optional[List]):
    if exclude is None:
        exclude = []

    if not isinstance(exclude, collections.Iterable):
        exclude = [exclude]

    for f in exclude:
        if callable(f):
            if f(experiment):
                return True
        else:
            if is_excluded_from_string(f, experiment):
                return True
    return False


def show(
        experiments: List[Experiment],
        targets=None,
        group_by=None,
        aggregator=None,
        sort: Optional[str] = None,
        exclude: Optional[List] = None,
        plot=None,
        options=None,
        export_filename=None,
):
    # Setup aggregator
    if aggregator == "count":
        aggregator = len
    elif aggregator == "mean" or aggregator is None:
        aggregator = mean
    else:
        raise RuntimeError("Unknown aggregator {}".format(aggregator))

    experiments = [e for e in experiments if not is_excluded(e, exclude)]

    if plot and len(targets) != 1:
        raise ValueError("Plotting requires exactly one target, {} given".format(len(targets)))

    group_by = group_by or (["id"] if plot else ["*"])

    if plot:
        targets = group_by[:1] + targets
        group_by = group_by[1:]

    if sort is not None:
        sort = sort.strip()
        if sort.startswith("-"):
            reverse = True
            sort = sort[1:]
        else:
            reverse = False

        experiments = [
            t[1] for t in sorted(zip(range(len(experiments)), experiments),
                                 key=lambda t: get_property(t[0], t[1], sort),
                                 reverse=reverse)
        ]

    groups = {}
    for i, experiment in enumerate(experiments):
        key = tuple([get_property(i, experiment, p) for p in group_by])
        if key not in groups:
            groups[key] = []
        groups[key].append(tuple([get_property(i, experiment, t) for t in targets]))

    keys = sorted(groups.keys())
    key_names = [["{}:{}".format(g, v) for g, v in zip(group_by, values)] for values in keys]

    if plot:
        scatter = ScatterData("", options)
        for n, k in zip(key_names, keys):
            name = ", ".join(map(str, n))
            sub_group = {}
            for r in groups[k]:
                if r[0] not in sub_group:
                    sub_group[r[0]] = []
                sub_group[r[0]].append(r[1])
            sub_keys = sorted(sub_group.keys())
            scatter.add_data(name, np.array(sub_keys), np.array([aggregator(sub_group[sk]) for sk in sub_keys]),
                             np.array([np.std(np.array(sub_group[sk])) for sk in sub_keys]))

        label_x = targets[0].capitalize()
        label_y = targets[1].capitalize()
        if targets[1] == "scores__mean":
            scatter.y_lim((0, 1))
        scatter.plot(export_filename,
                     lines=True, log_x=False, log_y=False, label_y=label_y, label_x=label_x, legend_pos="upper center")

    else:
        result_table = [
            ["{:.4f} (+/- {:.4f})".format(aggregator([r[i] for r in groups[k]]),
                                          np.std(np.array([r[i] for r in groups[k]])))
             for i, t in enumerate(targets)] for k in keys
        ]
        print(table(result_table, targets, list(zip(*key_names))))
