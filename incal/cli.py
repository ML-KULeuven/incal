from pywmi.smt_print import pretty_print

from .learn import LearnOptions


def main():
    formula, k, h = LearnOptions().execute_from_command_line("Learn SMT(LRA) theories from data")
    print("Learned formula (k={k}, h={h}): {f}".format(f=pretty_print(formula), k=k, h=h))

