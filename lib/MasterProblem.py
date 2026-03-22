from dataclasses import dataclass

import highspy

from lib.Dataset import Dataset
from lib.DemandScenario import DemandScenario


class MasterProblem:
    @dataclass
    class QualificationVariable:
        product: int
        factory: int

    @dataclass
    class WorkloadVariable:
        scenario: int
        product: int
        factory: int

    @dataclass
    class LostSalesVariable:
        scenario: int
        product: int

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.model = highspy.Highs()
        self.model.setMinimize()
        self.qualification_variables = {}
        self.workload_variables = {}
        self.lost_sales_variables = {}

        self.scenario = 0
        self.eta = self.model.addVariable(obj=1.0)
        for product in range(dataset.nmb_products):
            for factory in range(dataset.nmb_factories):
                self.add_qualification_variables(product=product, factory=factory,
                                                 obj=float(self.dataset.qualification_costs[product][factory]))

    def add_qualification_variables(self, product: int, factory: int, obj: float) -> None:
        self.lost_sales_variables[self.QualificationVariable(product=product, factory=factory)] = self.model.addBinary(
            obj=obj)

    def add_lost_sales_variable(self, product: int) -> None:
        self.lost_sales_variables[
            self.LostSalesVariable(scenario=self.scenario, product=product)] = self.model.addVariable()

    def add_workload_variable(self, product: int, factory: int) -> None:
        self.workload_variables[
            self.WorkloadVariable(scenario=self.scenario, product=product, factory=factory)] = self.model.addVariable()

    def add_scenario(self, demand_scenario: DemandScenario) -> None:
        # Workload variables.
        for product in range(self.dataset.nmb_products):
            for factory in range(self.dataset.nmb_factories):
                self.add_workload_variable(product=product, factory=factory)
            self.add_lost_sales_variable(product=product)

        # Flow constraints.
        for factory in range(self.dataset.nmb_factories):
            expr = self.model.expr()
            for product in range(self.dataset.nmb_products):
                expr += self.workload_variables[
                    self.WorkloadVariable(scenario=self.scenario, product=product, factory=factory)]
            self.model.addConstr(expr <= self.dataset.factory_capacities[factory])

        # Qualification constraints.
        for product in range(self.dataset.nmb_products):
            for factory in range(self.dataset.nmb_factories):
                x = self.workload_variables[
                    self.WorkloadVariable(scenario=self.scenario, product=product, factory=factory)]
                oq = self.qualification_variables[self.QualificationVariable(product=product, factory=factory)]
                self.model.addConstr(x <= float(demand_scenario.product_demands[product]) * oq)

        # Lost sales constraints.
        for product in range(self.dataset.nmb_products):
            expr = self.model.expr()
            expr += self.lost_sales_variables[self.LostSalesVariable(scenario=self.scenario, product=product)]
            for factory in range(self.dataset.nmb_factories):
                expr += self.workload_variables[
                    self.WorkloadVariable(scenario=self.scenario, product=product, factory=factory)]
            self.model.addConstr(expr == float(demand_scenario.product_demands[product]))

        # Objective function constraints.
        expr = self.model.expr()
        expr += self.eta
        for product in range(self.dataset.nmb_products):
            lost_sales_cost = float(self.dataset.lost_sales_costs[product])
            expr += -lost_sales_cost * self.lost_sales_variables[
                self.LostSalesVariable(scenario=self.scenario, product=product)]
        self.model.addConstr(expr >= 0.0)

        self.scenario += 1

    def solve(self):
        self.model.solve()
