class Learner(object):
    def learn(self, domain, data):
        raise NotImplementedError()

    @staticmethod
    def _convert(value):
        return float(value.constant_value())

    @staticmethod
    def _get_misclassification(data):
        true_count = 0
        for _, l in data:
            if l:
                true_count += 1
        return min(true_count, len(data) - true_count)
