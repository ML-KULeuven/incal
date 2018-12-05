class Observer(object):
    def observe(self, name, *args, **kwargs):
        raise NotImplementedError()


class DispatchObserver(Observer):
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)

    def observe(self, name, *args, **kwargs):
        for observer in self.observers:
            observer.observe(name, *args, **kwargs)


class SpecializedObserver(Observer):
    def observe(self, name, *args, **kwargs):
        instance_method_ref = getattr(self, "observe_{}".format(name))
        instance_method_ref(*args, **kwargs)
