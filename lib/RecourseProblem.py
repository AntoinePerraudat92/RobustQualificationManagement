from dataclasses import dataclass

import highspy
import numpy as np
from numpy.typing import NDArray

from lib.Dataset import Dataset
from lib.DemandScenario import DemandScenario


class RecourseProblem:
    @dataclass(frozen=True, eq=True)
    class WorkloadVariable:
        product: int
        factory: int

    @dataclass(frozen=True, eq=True)
    class LostSalesVariable:
        product: int

    @dataclass(frozen=True, eq=True)
    class QualificationConstraint:
        product: int
        factory: int

    @dataclass(frozen=True, eq=True)
    class LostSalesConstraint:
        product: int

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.model = highspy.Highs()
        self.model.silent()
        self.model.setMinimize()
        self.workload_variables = {}
        self.lost_sales_variables = {}

    def build(self, qualification_matrix: NDArray[np.int64], demand_scenario: DemandScenario):
        # Variables.
        for product in range(self.dataset.nmb_products):
            for factory in range(self.dataset.nmb_factories):
                self.workload_variables[
                    self.WorkloadVariable(product=product, factory=factory)] = self.model.addVariable()
            obj = float(self.dataset.lost_sales_costs[product])
            self.lost_sales_variables[self.LostSalesVariable(product=product)] = self.model.addVariable(obj=obj)

        # Flow constraints.
        for factory in range(self.dataset.nmb_factories):
            expr = self.model.expr()
            for product in range(self.dataset.nmb_products):
                expr += self.workload_variables[self.WorkloadVariable(product=product, factory=factory)]
            self.model.addConstr(expr <= self.dataset.factory_capacities[factory])

        # Qualification constraints.
        for product in range(self.dataset.nmb_products):
            for factory in range(self.dataset.nmb_factories):
                x = self.workload_variables[self.WorkloadVariable(product=product, factory=factory)]
                self.model.addConstr(x <= float(demand_scenario.product_demands[product]) * float(qualification_matrix[product][factory]))

        # Lost sales constraints.
        for product in range(self.dataset.nmb_products):
            expr = self.model.expr()
            expr += self.lost_sales_variables[self.LostSalesVariable(product=product)]
            for factory in range(self.dataset.nmb_factories):
                expr += self.workload_variables[self.WorkloadVariable(product=product, factory=factory)]
            self.model.addConstr(expr == float(demand_scenario.product_demands[product]))

    def solve(self):
        self.model.solve()

    def get_lost_sales(self) -> float:
        return float(np.sum([float(self.dataset.lost_sales_costs[product]) * self.model.val(self.lost_sales_variables[self.LostSalesVariable(product=product)]) for product in range(self.dataset.nmb_products)]))
