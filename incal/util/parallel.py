import subprocess
from multiprocessing.pool import Pool


def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.wait()
    print("[complete] {}".format(command))


def run_commands(commands):
    pool = Pool()
    pool.map(run_command, commands)
