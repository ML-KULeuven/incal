import json
import os
from fabric.api import run, env, execute, cd, local


def vary_synthetic_parameter(parameter_name, values, fixed_values, time_out=None, samples=None):
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
    }
    for key, value in fixed_values.items():
        if key not in default_values:
            raise RuntimeError("Found unknown parameter name {}".format(key))
        default_values[key] = value

    del default_values[parameter_name]

    config = {"fixed": default_values, "vary": parameter_name, "values": values}
    config_json = json.dumps(config)
    hash_value = hash(config_json)

    generation_dir_name = "synthetic/{}_{}".format(parameter_name, hash_value)
    output_dir_name = "output/{}".format(generation_dir_name)

    with(cd(env.smt_learning_root)):
        # Generate
        run("mkdir -p {}".format(generation_dir_name))
        with(cd("smtlearn")):
            commands = []
            for value in values:
                default_values[parameter_name] = value
                options = " ".join("--{} {}".format(name, val) for name, val in default_values.items())
                commands.append("echo python api.py generate ../{}/{} {}".format(generation_dir_name, value, options))
            commands.append("wait")
            run(" & ".join(commands))

        # Learn
        run("mkdir -p {}".format(output_dir_name))
        with(cd("smtlearn")):
            commands = []
            for value in values:
                command = "echo python experiments.py ../{input}/{val} "" ../{output}/{val} {bias}" \
                    .format(input=generation_dir_name, output=output_dir_name, val=value, bias=default_values["bias"])
                if time_out is not None:
                    command += " -t {}".format(time_out)
                commands.append(command)
            commands.append("wait")
            run(" & ".join(commands))

        # Combine
        """
        python api.py migrate ratio ../output/synthetic/hh/summary/ -d ../synthetic/hh/all -s 1000 -f
        python api.py migrate accuracy ../output/synthetic/hh/summary/ -d ../synthetic/hh/all -s 1000 -f
        """
        with cd(generation_dir_name):
            run("echo mkdir -p all")
            for value in values:
                run("echo cp {}/* all/".format(value))

        with cd("smtlearn"):
            run("echo python api.py combine ../{output}/summary {values} -p {output}/"
                .format(output=output_dir_name, values=" ".join(str(v) for v in values)))

            for migration in ["ratio", "accuracy"]:
                command = "echo python api.py migrate {migration} ../{output}/summary -d ../{input}/all" \
                    .format(output=output_dir_name, input=generation_dir_name, values=" ".join(str(v) for v in values),
                            migration=migration)
                if samples is not None:
                    command += " -s {}".format(samples)
                run(command)


def vary_h(time_out=None, samples=None):
    parameter_name = "half_spaces"
    values = [3, 4, 5, 6, 7, 8, 9, 10]
    fixed_values = {"data_sets": 100, "bool_count": 0, "real_count": 2, "k": 2, "literals": 3}
    vary_synthetic_parameter(parameter_name, values, fixed_values, time_out, samples)


if __name__ == "__main__":
    import authenticate

    authenticate.config()
    execute(vary_h, time_out=200, samples=1000)
