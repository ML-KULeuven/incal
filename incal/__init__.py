from pysmt.shortcuts import Real
from pywmi.domain import Density


class Formula(Density):
    def __init__(self, domain, support):
        super().__init__(domain, support, Real(1))

    @classmethod
    def from_state(cls, state: dict):
        density = Density.from_state(state)
        return cls(density.domain, density.support)
