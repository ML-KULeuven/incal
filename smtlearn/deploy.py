import json
import StringIO
import os
from os.path import join

import sys
from fabric.api import run, env, execute, cd, local, put, get, prefix
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
    config_json = json.dumps(config)
    if exp_name is None:
        exp_name = "h" + str(hash(config_json) + sys.maxsize + 1)

    print(config)

    generation_dir_name = "synthetic/{}/{}".format(parameter_name, exp_name)
    full_gen = join(env.smt_learning_root, generation_dir_name)

    output_dir_name = "output/{}".format(generation_dir_name)
    full_out = join(env.smt_learning_root, output_dir_name)

    full_code = join(env.smt_learning_root, "smtlearn")
    full_api = join(full_code, "api.py")
    full_exp = join(full_code, "experiments.py")

    python = env.python

    with(cd(env.smt_learning_root)):

        with(prefix("source {}".format(env.activate))):
            export_command = run("pysmt-install --env")

        # Generate
        if override or not files.exists("{}/config.json".format(full_gen)):
            run("mkdir -p {}".format(generation_dir_name))
            put(StringIO.StringIO(config_json), generation_dir_name + "/config.json")

            commands = []
            for value in values:
                default_values[parameter_name] = value
                options = " ".join("--{} {}".format(name, val) for name, val in default_values.items())
                command = "{python} {api} generate {input}/{val} {options}" \
                    .format(python=python, api=full_api, input=full_gen, val=value, options=options)
                commands.append("({} && {})".format(export_command, command))
            commands.append("wait")

            run(" & ".join(commands))

        # Learn
        if override or not files.exists("{}/config.json".format(full_out)):
            run("mkdir -p {}".format(output_dir_name))
            put(StringIO.StringIO(config_json), output_dir_name + "/config.json")

            commands = []
            for value in values:
                options = " ".join("--{} {}".format(name, val) for name, val in learner_settings.items())
                command = "{python} {exp} {input}/{val} \"\" {output}/{val} {options}" \
                    .format(python=python, exp=full_exp, input=full_gen, output=full_out, val=value, options=options)
                if time_out is not None:
                    command += " -t {}".format(time_out)
                commands.append("({} && {})".format(export_command, command))
            commands.append("wait")

            run(" & ".join(commands))

        # Combine
        if override or not files.exists("{}/summary".format(full_out)):
            with cd(generation_dir_name):
                run("mkdir -p all")
                for value in values:
                    run("cp {}/* all/".format(value))

            with(prefix(export_command)):
                run("{python} {api} combine {output}/summary {values} -p {output}/"
                    .format(python=python, api=full_api, output=full_out, values=" ".join(str(v) for v in values)))

            for migration in ["ratio", "accuracy"]:
                command = "{python} {api} migrate {migration} {output}/summary -d {input}/all" \
                    .format(output=full_out, input=full_gen, values=" ".join(str(v) for v in values), api=full_api,
                            migration=migration, python=python)
                if samples is not None:
                    command += " -s {}".format(samples)
                with(prefix(export_command)):
                    run(command)

        local_root = os.path.dirname(os.path.dirname(__file__))
        if override or not os.path.exists(join(local_root, generation_dir_name)):
            get(generation_dir_name, "{}/{}".format(local_root, os.path.dirname(generation_dir_name)))
        if override or not os.path.exists(join(local_root, output_dir_name)):
            get(output_dir_name, "{}/{}".format(local_root, os.path.dirname(output_dir_name)))


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
    values = [3, 4]
    fixed_values = {"data_sets": 10, "bool_count": 0, "real_count": 2, "k": 2, "literals": 3}

    learner = {"bias": "cnf", "selection": "random"}
    vary_synthetic_parameter(parameter_name, values, fixed_values, learner, time_out, samples)

    # learner["selection"] = "dt_weighted"
    # vary_synthetic_parameter(parameter_name, values, fixed_values, learner, time_out, samples)


if __name__ == "__main__":
    import authenticate

    authenticate.config()
    execute(vary_h_simple, time_out=200, samples=1000)
