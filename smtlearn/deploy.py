import json
import StringIO
import os
from os.path import join, dirname

import sys
from fabric.api import run, env, execute, cd, local, put, get, prefix, lcd
from fabric.contrib import files


def vary_synthetic_parameter(parameter_name, values, fixed_values, learner_settings, time_out=None, samples=None,
                             exp_name=None, override=False):
    default_values = {
        "data_sets": 10,
        "bool_count": 2,
        "real_count": 2,
        "bias": "cnf",
        "k": 3,
        "literals": 4,
        "half_spaces": 7,
        "samples": 1000,
        "ratio": 90,
        "errors": 0,
    }
    for key, value in fixed_values.items():
        if key not in default_values:
            raise RuntimeError("Found unknown parameter name {}".format(key))
        default_values[key] = value

    del default_values[parameter_name]

    config = {"fixed": default_values, "vary": parameter_name, "values": values, "learner": learner_settings}
    if exp_name is None:
        exp_name = "h" + str(hash(json.dumps(config)) + sys.maxsize + 1)

    print(config)

    exp_path = join("synthetic", parameter_name, exp_name)
    local_root = dirname(dirname(__file__))
    full_gen = join(local_root, exp_path)
    full_out = join(local_root, "output", exp_path)
    full_code = join(local_root, "smtlearn")
    full_api = join(full_code, "api.py")
    full_exp = join(full_code, "experiments.py")

    # Generate
    gen_config = join(full_gen, "config.json")
    if override or not os.path.exists(gen_config):
        local("mkdir -p {}".format(full_gen))

        with open(gen_config, "w") as f:
            json.dump(config, f)

        commands = []
        for value in values:
            default_values[parameter_name] = value
            options = " ".join("--{} {}".format(name, val) for name, val in default_values.items())
            command = "python {api} generate {input}/{val} {options}" \
                .format(api=full_api, input=full_gen, val=value, options=options)
            commands.append(command)
        commands.append("wait")

        local(" & ".join(commands))

    # Learn
    out_config = join(full_out, "config.json")
    if override or not os.path.exists(out_config):
        local("mkdir -p {}".format(full_out))

        with open(out_config, "w") as f:
            json.dump(config, f)

        commands = []
        for value in values:
            options = " ".join("--{} {}".format(name, val) for name, val in learner_settings.items())
            command = "python {exp} {input}/{val} \"\" {output}/{val} {options}" \
                .format(exp=full_exp, input=full_gen, output=full_out, val=value, options=options)
            if time_out is not None:
                command += " -t {}".format(time_out)
            commands.append(command)
        commands.append("wait")

        local(" & ".join(commands))

    # Combine
    if override or not os.path.exists(join(full_out, "summary")):
        with lcd(full_gen):
            local("mkdir -p all")
            for value in values:
                local("cp {}/* all/".format(value))

        local("python {api} combine {output}/summary {values} -p {output}/"
            .format(api=full_api, output=full_out, values=" ".join(str(v) for v in values)))

        for migration in ["ratio", "accuracy"]:
            command = "python {api} migrate {migration} {output}/summary -d {input}/all" \
                .format(output=full_out, input=full_gen, values=" ".join(str(v) for v in values), api=full_api,
                        migration=migration)
            if samples is not None:
                command += " -s {}".format(samples)
            local(command)


def vary_h(time_out=None, samples=None, override=False):
    parameter = "half_spaces"
    values = [3, 4, 5, 6, 7, 8, 9, 10]
    fixed_values = {"data_sets": 100, "bool_count": 0, "real_count": 2, "k": 2, "literals": 3}

    learner = {"bias": "cnf", "selection": "random"}
    vary_synthetic_parameter(parameter, values, fixed_values, learner, time_out, samples, "standard", override)

    learner["selection"] = "dt_weighted"
    vary_synthetic_parameter(parameter, values, fixed_values, learner, time_out, samples, "dt", override)


def vary_h_simple(time_out=None, samples=None):
    parameter_name = "half_spaces"
    values = [3, 4, 5, 6, 7, 8]
    fixed_values = {"data_sets": 10, "bool_count": 0, "real_count": 2, "k": 2, "literals": 3}

    learner = {"bias": "cnf", "selection": "random"}
    vary_synthetic_parameter(parameter_name, values, fixed_values, learner, time_out, samples, "small_standard")

    learner["selection_size"] = 1
    vary_synthetic_parameter(parameter_name, values, fixed_values, learner, time_out, samples, "small_standard_single")

    learner["selection_size"] = 20
    vary_synthetic_parameter(parameter_name, values, fixed_values, learner, time_out, samples, "small_standard_20")

    learner["selection"] = "dt_weighted"
    learner["selection_size"] = 1
    vary_synthetic_parameter(parameter_name, values, fixed_values, learner, time_out, samples, "small_dt_1")

    learner["selection"] = "dt"
    learner["selection_size"] = 1
    vary_synthetic_parameter(parameter_name, values, fixed_values, learner, time_out, samples, "small_sdt_1")


if __name__ == "__main__":
    import authenticate

    authenticate.config()
    execute(vary_h_simple, time_out=200, samples=1000)
