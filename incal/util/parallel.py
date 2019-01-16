import os
import subprocess
from multiprocessing.pool import Pool


def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.wait()
    print("[complete] {}".format(command))


def run_commands(commands, processes=None):
    pool = Pool(processes=processes)
    pool.map(run_command, commands)
