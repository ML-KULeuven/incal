import subprocess
from multiprocessing.pool import Pool


def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.wait()
    print("completed")


def run_commands(commands):
    pool = Pool(processes=4)
    pool.map(run_command, commands)
