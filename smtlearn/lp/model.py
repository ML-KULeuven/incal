from pysmt.fnode import FNode
from pywmi import Domain
from typing import List


class Model(object):
    def __init__(self, domain: Domain, objective: FNode, constraints: List[FNode], minimize: bool=True, name=None):
        self.domain = domain
        self.objective = objective
        self.constraints = constraints
        self.minimize = minimize
        self.name = name
