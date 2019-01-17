import subprocess
from multiprocessing.pool import Pool
from subprocess import TimeoutExpired


def run_command(args):
    command, time_out = args
    process = subprocess.Popen(command, shell=True)
    try:
        process.wait(timeout=time_out)
        print("[complete] {}".format(command))
    except TimeoutExpired:
        process.kill()
        


def run_commands(commands, processes=None, time_out=None):
    pool = Pool(processes=processes)
    commands = [(command, time_out) for command in commands]
    pool.map(run_command, commands)
