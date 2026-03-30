from dataclasses import dataclass

import pyomo.environ as pyo
import numpy as np
from numpy.typing import NDArray

from src.data_model.Dataset import Dataset
from src.data_model.DemandScenario import DemandScenario


class MasterProblem:
    @dataclass(frozen=True, eq=True)
    class WorkloadVariable:
        product: int
        factory: int

    @dataclass(frozen=True, eq=True)
    class LostSalesVariable:
        product: int

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.model = pyo.ConcreteModel()
        self.solver = pyo.SolverFactory('appsi_highs')
        self.model.products = pyo.Set(initialize=[product for product in range(self.dataset.nmb_products)])
        self.model.factories = pyo.Set(initialize=[factory for factory in range(self.dataset.nmb_factories)])

        # Add variables common to all scenarios.
        self.model.eta = pyo.Var()
        self.model.qualification_variables = pyo.Var(self.model.products, self.model.factories, within=pyo.Binary)
        self.model.workload_variables = pyo.VarList(within=pyo.PositiveReals)
        self.model.lost_sales_variables = pyo.VarList(within=pyo.PositiveReals)
        self.model.constraints = pyo.ConstraintList()

        # Objective function.
        def objective_function_rule(model):
            return model.eta + sum(
                self.dataset.qualification_costs[product][factory] * model.qualification_variables[product, factory]
                for product in model.products for factory in model.factories)

        self.model.objective = pyo.Objective(rule=objective_function_rule, sense=pyo.minimize)

    def add_scenario(self, demand_scenario: DemandScenario) -> None:
        workload_variables_dict = dict()
        lost_sale_variables_dict = dict()
        # Variables.
        for product in self.model.products:
            for factory in self.model.factories:
                workload_variables_dict[
                    self.WorkloadVariable(product=product, factory=factory)] = self.model.workload_variables.add()
            lost_sale_variables_dict[self.LostSalesVariable(product=product)] = self.model.lost_sales_variables.add()

        # Capacity constraints.
        for factory in self.model.factories:
            capacity = self.dataset.factory_capacities[factory]
            self.model.constraints.add(sum(
                workload_variables_dict[self.WorkloadVariable(product=product, factory=factory)] for product in
                self.model.products) <= float(capacity))

        # Qualification constraints.
        for product in self.model.products:
            for factory in self.model.factories:
                workload_variable = workload_variables_dict[self.WorkloadVariable(product=product, factory=factory)]
                if int(self.dataset.qualification_matrix[product][factory]) == 1:
                    demand = float(demand_scenario.product_demands[product])
                    self.model.constraints.add(
                        workload_variable <= demand * self.model.qualification_variables[product, factory])
                else:
                    self.model.constraints.add(workload_variable <= 0.0)

        # Flow constraints.
        for product in self.model.products:
            demand = float(demand_scenario.product_demands[product])
            self.model.constraints.add(lost_sale_variables_dict[self.LostSalesVariable(product=product)] + sum(
                workload_variables_dict[self.WorkloadVariable(product=product, factory=factory)] for factory in
                self.model.factories) == demand)

        # Objective function constraint.
        self.model.constraints.add(self.model.eta >= sum(
            self.dataset.lost_sales_costs[product] * lost_sale_variables_dict[self.LostSalesVariable(product=product)]
            for product in
            self.model.products))

    def solve(self):
        results = self.solver.solve(self.model)
        return (results.solver.termination_condition == pyo.TerminationCondition.optimal
                or results.solver.termination_condition == pyo.TerminationCondition.feasible)

    def get_objective_function(self) -> float:
        return pyo.value(self.model.objective)

    def get_qualification_matrix(self) -> NDArray[np.float64]:
        return np.array(
            [[self.model.qualification_variables[product, factory].value for factory in
              self.model.factories] for
             product in self.model.products], dtype=np.float64)

    def get_qualification_decision(self, product: int, factory: int) -> int:
        return self.model.qualification_variables[product, factory].value

    def get_qualification_costs(self) -> float:
        return pyo.value(self.model.objective) - self.model.eta.value

    def get_worst_case_lost_sales(self) -> float:
        return self.model.eta.value
