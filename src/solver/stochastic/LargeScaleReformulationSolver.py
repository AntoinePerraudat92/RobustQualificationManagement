import numpy as np
import pyomo.environ as pyo
from numpy.typing import NDArray

from src.data_model.Dataset import Dataset
from src.data_model.DemandScenario import DemandScenario


class LargeScaleReformulationSolver:

    def __init__(self, dataset: Dataset, demand_scenarios: list[DemandScenario], w: float = 0.5, alpha: float = 0.90):
        self.dataset = dataset
        self.demand_scenarios = demand_scenarios
        self.model = pyo.ConcreteModel()
        self.solver = pyo.SolverFactory('appsi_highs')
        nmb_scenarios = len(demand_scenarios)
        self.proba_per_scenario = 1.0 / float(nmb_scenarios)
        self.w = w
        self.model.scenarios = pyo.Set(initialize=[scenario for scenario in range(nmb_scenarios)])
        self.model.products = pyo.Set(initialize=[product for product in range(self.dataset.nmb_products)])
        self.model.factories = pyo.Set(initialize=[factory for factory in range(self.dataset.nmb_factories)])

        self.model.qualification_variables = pyo.Var(self.model.products, self.model.factories, within=pyo.Binary)
        self.model.workload_variables = pyo.Var(self.model.scenarios, self.model.products, self.model.factories,
                                                within=pyo.NonNegativeReals)
        self.model.lost_sale_variables = pyo.Var(self.model.scenarios, self.model.products, within=pyo.NonNegativeReals)

        self.model.thetas = pyo.Var(self.model.scenarios, within=pyo.PositiveReals)
        self.model.cvar = pyo.Var(within=pyo.Reals)
        self.model.betas = pyo.Var(self.model.scenarios, within=pyo.PositiveReals)
        self.model.eta = pyo.Var(within=pyo.Reals)

        # Objective function.
        def objective_function_rule(model):
            scalar = (1.0 - self.w) * self.proba_per_scenario
            return sum(
                self.dataset.qualification_costs[product][factory] * model.qualification_variables[product, factory]
                for product in model.products for factory in model.factories) + sum(
                scalar * model.thetas[scenario] for scenario in model.scenarios) + self.w * model.cvar

        self.model.objective = pyo.Objective(rule=objective_function_rule, sense=pyo.minimize)

        # Capacity constraints.
        def capacity_constraints_rule(model, scenario, factory):
            return sum(model.workload_variables[scenario, product, factory] for product in model.products) <= float(
                self.dataset.factory_capacities[factory])

        self.model.flow_constraint = pyo.Constraint(self.model.scenarios, self.model.factories,
                                                    rule=capacity_constraints_rule)

        # Lost sales constraints.
        def lost_sales_constraints_rule(model, scenario, product):
            return model.lost_sale_variables[scenario, product] + sum(
                model.workload_variables[scenario, product, factory] for factory in model.factories) == \
                self.demand_scenarios[scenario].product_demands[product]

        self.model.lost_sales_constraint = pyo.Constraint(self.model.scenarios, self.model.products,
                                                          rule=lost_sales_constraints_rule)

        # Qualification constraints.
        def qualification_constraint_rule(model, scenario, product, factory):
            demand = self.demand_scenarios[scenario].product_demands[product]
            if int(self.dataset.qualification_matrix[product][factory]) == 1:
                return model.workload_variables[scenario, product, factory] <= \
                    demand * model.qualification_variables[product, factory]
            else:
                return model.workload_variables[scenario, product, factory] <= 0

        self.model.qualification_constraints = pyo.Constraint(self.model.scenarios, self.model.products,
                                                              self.model.factories,
                                                              rule=qualification_constraint_rule)

        # Expected value constraints.
        def expected_value_constraint_rule(model, scenario):
            return model.thetas[scenario] >= sum(
                self.dataset.lost_sales_costs[product] * model.lost_sale_variables[scenario, product] for product in
                model.products)

        self.model.expected_value_constraints = pyo.Constraint(self.model.scenarios,
                                                               rule=expected_value_constraint_rule)

        # CVaR constraints.
        def beta_constraint_rule(model, scenario):
            return model.betas[scenario] >= model.thetas[scenario] - model.eta

        self.model.beta_constraints = pyo.Constraint(self.model.scenarios, rule=beta_constraint_rule)

        def cvar_constraint_rule(model):
            scalar = 1.0 / (1.0 - alpha) * self.proba_per_scenario
            return model.eta + sum(scalar * model.betas[scenario] for scenario in model.scenarios) <= model.cvar

        self.model.cvar_constraints = pyo.Constraint(rule=cvar_constraint_rule)

    def solve(self) -> bool:
        results = self.solver.solve(self.model, tee=True)
        return (results.solver.termination_condition == pyo.TerminationCondition.optimal
                or results.solver.termination_condition == pyo.TerminationCondition.feasible)

    def get_qualification_decision(self, product: int, factory: int) -> int:
        return self.model.qualification_variables[product, factory].value

    def get_qualification_matrix(self) -> NDArray[np.float64]:
        return np.array(
            [[self.get_qualification_decision(product=product, factory=factory) for factory in self.model.factories] for
             product in self.model.products], dtype=np.float64)

    def get_qualification_costs(self) -> float:
        lost_sales = (1.0 - self.w) * self.get_expected_lost_sales()
        return pyo.value(self.model.objective) - lost_sales - self.w * self.get_cvar()

    def get_objective_function(self) -> float:
        return pyo.value(self.model.objective)

    def get_expected_lost_sales(self) -> float:
        return sum(self.proba_per_scenario * self.model.thetas[scenario].value for scenario in self.model.scenarios)

    def get_cvar(self) -> float:
        return self.model.cvar.value
