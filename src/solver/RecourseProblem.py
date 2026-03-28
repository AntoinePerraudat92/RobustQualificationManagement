import numpy as np
import pyomo.environ as pyo
from numpy.typing import NDArray

from src.data_model.Dataset import Dataset
from src.data_model.DemandScenario import DemandScenario


class RecourseProblem:
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.model = pyo.ConcreteModel()
        self.solver = pyo.SolverFactory('appsi_highs')
        self.model.products = pyo.Set(initialize=[product for product in range(self.dataset.nmb_products)])
        self.model.factories = pyo.Set(initialize=[factory for factory in range(self.dataset.nmb_factories)])
        self.model.workload_variables = pyo.Var(self.model.products, self.model.factories, domain=pyo.NonNegativeReals)
        self.model.lost_sales_variables = pyo.Var(self.model.products, domain=pyo.NonNegativeReals)

        self.model.demand = pyo.Param(self.model.products, mutable=True)
        self.model.qualification_rhs = pyo.Param(self.model.products, self.model.factories, mutable=True)

        # Objective function.
        def objective_function_rule(model):
            return sum(self.dataset.lost_sales_costs[product] * model.lost_sales_variables[product] for product in model.products)
        self.model.objective = pyo.Objective(rule=objective_function_rule, sense=pyo.minimize)

        # Capacity constraints.
        def capacity_constraints_rule(model, factory):
            return sum(model.workload_variables[product, factory] for product in self.model.products) <= float(
                self.dataset.factory_capacities[factory])
        self.model.flow_constraint = pyo.Constraint(self.model.factories, rule=capacity_constraints_rule)

        # Qualification constraints.
        def qualification_constraints_rule(model, product, factory):
            return model.workload_variables[product, factory] <= self.model.qualification_rhs[product, factory]
        self.model.qualification_constraint = pyo.Constraint(self.model.products, self.model.factories,
                                                             rule=qualification_constraints_rule)

        # Lost sales constraints.
        def lost_sales_constraints_rule(model, product):
            return model.lost_sales_variables[product] + sum(
                model.workload_variables[product, factory] for factory in self.model.factories) == self.model.demand[product]
        self.model.lost_sales_constraint = pyo.Constraint(self.model.products, rule=lost_sales_constraints_rule)

    def solve(self, qualification_matrix: NDArray[np.int64], demand_scenario: DemandScenario) -> bool:
        for product in self.model.products:
            self.model.demand[product] = demand_scenario.product_demands[product]
            for factory in self.model.factories:
                self.model.qualification_rhs[product, factory] = demand_scenario.product_demands[product] * qualification_matrix[product][factory]
        results = self.solver.solve(self.model)
        return (results.solver.termination_condition == pyo.TerminationCondition.optimal
                or results.solver.termination_condition == pyo.TerminationCondition.feasible)

    def get_lost_sales(self) -> float:
        return pyo.value(self.model.objective)
