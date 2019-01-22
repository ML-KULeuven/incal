from pywmi.smt_print import pretty_print

from incal.learn import LearnOptions
from pywmi import evaluate
from pywmi.sample import uniform

from incal.experiments.examples import simple_checker_problem, checker_problem
from incal.violations.core import RandomViolationsStrategy

from incal.violations.virtual_data import OneClassStrategy

from incal.k_cnf_smt_learner import KCnfSmtLearner

from incal.parameter_free_learner import learn_bottom_up

# from incal.observe.inc_logging import LoggingObserver


def main():
    domain, formula, name = checker_problem()
    thresholds = {v: 0.1 for v in domain.real_vars}
    data = uniform(domain, 1000)
    labels = evaluate(domain, formula, data)
    data = data[labels == 1]
    labels = labels[labels == 1]

    def learn_inc(_data, _labels, _i, _k, _h):
        strategy = OneClassStrategy(RandomViolationsStrategy(1), thresholds)
        learner = KCnfSmtLearner(_k, _h, strategy, "mvn")
        initial_indices = LearnOptions.initial_random(20)(list(range(len(data))))
        # learner.add_observer(LoggingObserver(None, _k, _h, None, True))
        return learner.learn(domain, _data, _labels, initial_indices)

    (new_data, new_labels, formula), k, h = learn_bottom_up(data, labels, learn_inc, 1, 1, 1, 1, None, None)
    print("Learned CNF(k={}, h={}) formula {}".format(k, h, pretty_print(formula)))
    print("Data-set grew from {} to {} entries".format(len(labels), len(new_labels)))


if __name__ == "__main__":
    main()
