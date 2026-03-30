import numpy as np
import pyomo.environ as pyo
from numpy.typing import NDArray

from src.data_model.Dataset import Dataset
from src.data_model.DemandScenario import DemandScenario


class DualRecourseProblem:

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.model = pyo.ConcreteModel()
        self.solver = pyo.SolverFactory('appsi_highs')
        self.model.products = pyo.Set(initialize=[product for product in range(self.dataset.nmb_products)])
        self.model.factories = pyo.Set(initialize=[factory for factory in range(self.dataset.nmb_factories)])
        self.model.W = pyo.Var(self.model.products, within=pyo.Reals)
        self.model.Y = pyo.Var(self.model.factories, within=pyo.NonNegativeReals)
        self.model.Z = pyo.Var(self.model.products, self.model.factories, within=pyo.NonNegativeReals)

        self.model.demand = pyo.Param(self.model.products, within=pyo.Reals, mutable=True)
        self.model.q = pyo.Param(self.model.products, self.model.factories, within=pyo.Reals, mutable=True)

        # Objective function.
        def objective_function_rule(model):
            return (sum(model.demand[product] * model.W[product] for product in model.products)
                    + sum(-self.dataset.factory_capacities[factory] * model.Y[factory] for factory in model.factories)
                    + sum(
                        -model.q[product, factory] * model.Z[product, factory] for product in model.products for factory
                        in model.factories))

        self.model.objective = pyo.Objective(rule=objective_function_rule, sense=pyo.maximize)

        # Dual constraints linked to workload variables.
        def dual_workload_variables_rule(model, product, factory):
            return -model.Z[product, factory] - model.Y[factory] + model.W[product] <= 0

        self.model.dual_workload_variables_constraint = pyo.Constraint(self.model.products, self.model.factories,
                                                                       rule=dual_workload_variables_rule)

        # Dual constraints linked to lost sales.
        def dual_lost_sale_variables_rule(model, product):
            return model.W[product] <= self.dataset.lost_sales_costs[product]

        self.model.dual_lost_sale_variables_constraint = pyo.Constraint(self.model.products,
                                                                        rule=dual_lost_sale_variables_rule)

    def solve(self, qualification_matrix: NDArray[np.int64], demand_scenario: DemandScenario) -> bool:
        for product in self.model.products:
            self.model.demand[product] = demand_scenario.product_demands[product]
            for factory in self.model.factories:
                self.model.q[product, factory] = demand_scenario.product_demands[product] * \
                                                 qualification_matrix[product][factory]
        results = self.solver.solve(self.model)
        return (results.solver.termination_condition == pyo.TerminationCondition.optimal
                or results.solver.termination_condition == pyo.TerminationCondition.feasible)

    def get_lost_sales(self) -> float:
        return pyo.value(self.model.objective)

    def get_benders_cut_constant(self) -> float:
        return sum(self.model.demand[product].value * self.model.W[product].value for product in self.model.products) + sum(
            -self.dataset.factory_capacities[factory] * self.model.Y[factory].value for factory in self.model.factories)

    def get_benders_cut_coefficients(self) -> NDArray[np.float64]:
        return np.array([[self.model.Z[product, factory].value for factory in self.model.factories] for product in self.model.products], dtype=np.float64)
