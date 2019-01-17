import json
import os
import time
from typing import Union, Tuple, Any, Dict, List, Optional


class Option(object):
    def __init__(self, name, from_type=None, default_from=None, converter=None, default=None, arg_name=None):
        self.name = name
        self.from_type = from_type
        self.default_from = default_from
        self.converter = converter
        self.default = default
        self.arg_name = arg_name or name
        if converter is not None and default_from is not None and default is None:
            self.default = self.convert(default_from)

    def convert(self, value):
        if value is None:
            return self.default
        elif self.converter:
            if isinstance(self.from_type, tuple):
                parts = value.split(":")
                if len(parts) != len(self.from_type):
                    raise RuntimeError("Could not parse arguments for option {}, got: {}".format(self.name, value))
                value = tuple(t(v) for t, v in zip(self.from_type, parts))

            if isinstance(value, tuple):
                return self.converter(*value)
            else:
                return self.converter(value)
        else:
            return value


class Options(object):
    def __init__(self, callback=None):
        self.options = dict()
        self.values = dict()
        self.original_values = dict()
        self.callback = callback

    def add_option(self, name, from_type=None, default_from=None, converter=None, default=None, arg_name=None):
        if isinstance(default_from, tuple):
            default_from = ":".join(str(e) for e in default_from)
        self.options[name] = Option(name, from_type, default_from, converter, default, arg_name)

    def set_values(self, convert=True, **kwargs):
        for key, value in kwargs.items():
            self.set_value(key, value, convert)

    @staticmethod
    def convert_dict(**kwargs):
        def convert(*args):
            if args[0] in kwargs:
                if len(args) > 1:
                    return kwargs[args[0]](*args[1:])
                return kwargs[args[0]]
            raise RuntimeError("Unknown option {}, should be one of: {}".format(args[0], list(kwargs.keys())))

        return convert

    def set_value(self, name, value, convert=True):
        self.original_values[name] = value
        if convert:
            self.values[name] = self.options[name].convert(value)
        else:
            self.values[name] = value

    def __setattr__(self, key, value):
        if key in ["options", "values", "original_values", "callback"] or key.startswith("__"):
            return super().__setattr__(key, value)
        self.set_value(key, value)

    def __getattr__(self, item):
        if item in ["options", "values", "original_values", "callback"] or item.startswith("__"):
            return super().__getattr__(item)
        return self.values[item] if item in self.values else self.options[item].default

    def add_arguments(self, parser):
        for o_name, option in self.options.items():
            parser.add_argument(
                "--{}".format(option.name),
                type=option.from_type if not isinstance(option.from_type, tuple) else str,
                default=option.default_from
            )

    def parse_arguments(self, args):
        for o_name, option in self.options.items():
            self.set_value(option.name, getattr(args, option.name))

    def print_arguments(self):
        return " ".join("--{} {}".format(name, o_value) for name, o_value in self.original_values.items()
                        if o_value is not None)

    def call(self, timed=False) -> Union[Tuple[Any, float], Any]:
        def make_call():
            return self.callback(**{self.options[o_name].arg_name: value for o_name, value in self.values.items()})

        if timed:
            start_time = time.time()
            result = make_call()
            duration = time.time() - start_time
            return result, duration
        else:
            return make_call()

    def execute_from_command_line(self, description: str=None, timed: bool=False) -> Union[Tuple[Any, float], Any]:
        import argparse
        parser = argparse.ArgumentParser(description=description)
        self.add_arguments(parser)
        self.parse_arguments(parser.parse_args())
        return self.call(timed)

    def copy(self):
        options = self.make_copy()
        options.options = dict(self.options)
        options.values = dict(self.values)
        options.original_values = dict(self.original_values)
        return options

    def make_copy(self):
        return Options(self.callback)

    def export_to_dict(self):
        return dict(self.original_values)

    def import_from_dict(self, values_dict):
        self.set_values(True, **values_dict)


class Results(Options):
    @staticmethod
    def make_converter(converter):
        def convert(result, duration):
            return converter(result)
        return convert

    def add_result(self, name, converter):
        self.add_option(name, converter=Results.make_converter(converter))

    def add_duration(self, name="duration"):
        def convert(result, duration):
            return duration
        self.add_option(name, converter=convert)

    def export_to_dict(self):
        return dict(self.values)

    def import_from_dict(self, values_dict):
        self.set_values(False, **values_dict)


class Experiment(object):
    def __init__(self, parameters: Options, results: Options, config: Optional[Options]=None, import_handler=None):
        self.parameters = parameters
        self.results = results
        self.config = config
        self.import_handler = import_handler
        self.derived = dict()
        self.imported_from_file = None

    def register_derived(self, name, callback):
        self.derived[name] = callback

    def import_from_command_line(self):
        import argparse
        parser = argparse.ArgumentParser()
        self.parameters.add_arguments(parser)
        if self.config:
            self.config.add_arguments(parser)
        args = parser.parse_args()
        self.parameters.parse_arguments(args)
        if self.config:
            self.config.parse_arguments(args)

    def execute_from_command_line(self):
        self.import_from_command_line()
        self.execute()

    def execute(self):
        result = self.parameters.call(timed=True)
        for o_name in self.results.options:
            self.results.set_value(o_name, result)

    def export_to_dict(self):
        return {"parameters": self.parameters.export_to_dict(), "results": self.results.export_to_dict(),
                "config": self.config.export_to_dict() if self.config else None}

    def save(self, filename):
        with open(filename, "w") as ref:
            json.dump(self.export_to_dict(), ref)

    def import_from_dict(self, values_dict):
        parameters_dict, results_dict, config_dict = (values_dict[k] for k in ["parameters", "results", "config"])
        if self.import_handler is not None:
            self.import_handler(parameters_dict, results_dict, config_dict)
        self.parameters.import_from_dict(parameters_dict)
        self.results.import_from_dict(results_dict)
        if self.config and config_dict:
            self.config.import_from_dict(config_dict)

    def load(self, filename):
        with open(filename, "r") as ref:
            self.import_from_dict(json.load(ref))
        self.imported_from_file = os.path.realpath(filename)
        return self



