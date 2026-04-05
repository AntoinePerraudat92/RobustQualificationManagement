import math
import numpy as np
import pyomo.environ as pyo
from numpy.typing import NDArray
from tqdm import tqdm

from src.data_model.Dataset import Dataset
from src.data_model.DemandScenario import DemandScenario
from src.solver.stochastic.DualRecourseProblem import DualRecourseProblem
from src.util.gap_util import compute_gap


class BendersDecompositionSolver:

    def __init__(self, dataset: Dataset, demand_scenarios: list[DemandScenario]):
        self.dataset = dataset
        self.demand_scenarios = demand_scenarios
        self.model = pyo.ConcreteModel()
        self.solver = pyo.SolverFactory('appsi_highs')
        nmb_scenarios = len(demand_scenarios)
        proba_per_scenario = 1.0 / float(nmb_scenarios)
        self.model.scenarios = pyo.Set(initialize=[scenario for scenario in range(nmb_scenarios)])
        self.model.products = pyo.Set(initialize=[product for product in range(self.dataset.nmb_products)])
        self.model.factories = pyo.Set(initialize=[factory for factory in range(self.dataset.nmb_factories)])

        self.model.qualification_variables = pyo.Var(self.model.products, self.model.factories, within=pyo.Binary)
        self.model.thetas = pyo.Var(self.model.scenarios, within=pyo.PositiveReals)
        self.model.cuts = pyo.ConstraintList()

        # Objective function.
        def objective_function_rule(model):
            return sum(
                self.dataset.qualification_costs[product][factory] * model.qualification_variables[product, factory]
                for product in model.products for factory in model.factories) + sum(
                proba_per_scenario * model.thetas[scenario] for scenario in model.scenarios)

        self.model.objective = pyo.Objective(rule=objective_function_rule, sense=pyo.minimize)

        # Qualification constraints.
        def qualification_constraint_rule(model, product, factory):
            return model.qualification_variables[product, factory] <= self.dataset.qualification_matrix[product][
                factory]

        self.model.qualification_constraints = pyo.Constraint(self.model.products, self.model.factories,
                                                              rule=qualification_constraint_rule)

    def run(self) -> bool:
        results = self.solver.solve(self.model)
        return (results.solver.termination_condition == pyo.TerminationCondition.optimal
                or results.solver.termination_condition == pyo.TerminationCondition.feasible)

    def get_qualification_decision(self, product: int, factory: int) -> int:
        return self.model.qualification_variables[product, factory].value

    def get_qualification_matrix(self) -> NDArray[np.float64]:
        return np.array(
            [[self.get_qualification_decision(product=product, factory=factory) for factory in self.model.factories] for
             product in self.model.products], dtype=np.float64)

    def get_qualification_costs(self) -> float:
        return pyo.value(self.model.objective) - self.get_lost_sales()

    def get_lost_sales(self) -> float:
        proba_per_scenario = 1.0 / float(len(self.demand_scenarios))
        return sum(proba_per_scenario * self.model.thetas[scenario].value for scenario in self.model.scenarios)

    def add_benders_cut(self, scenario: int, recourse_problem: DualRecourseProblem):
        constant = recourse_problem.get_benders_cut_constant()
        coefficients = recourse_problem.get_benders_cut_coefficients()
        self.model.cuts.add(self.model.thetas[scenario] >= constant + sum(
            -coefficients[product][factory] * self.model.qualification_variables[product, factory] for product in
            self.model.products for factory in self.model.factories))

    def solve(self):
        recourse_problem: DualRecourseProblem = DualRecourseProblem(self.dataset)
        lb: float = -math.inf
        ub: float = math.inf
        gap = compute_gap(lb=lb, ub=ub)
        print(f"lb: {lb}, ub: {ub}, gap: {gap} (%)")
        nmb_scenarios = len(self.demand_scenarios)
        proba_per_scenario = 1.0 / float(len(self.demand_scenarios))
        # We solve the master problem for any scenario to have meaningful qualification decisions for the recourse problem.
        self.run()
        while gap > 1E-4:
            qualification_matrix: NDArray[np.float64] = self.get_qualification_matrix()
            lost_sales: NDArray[np.float64] = np.zeros(shape=(nmb_scenarios,), dtype=np.float64)

            print('Solving recourse problems...')
            for scenario in tqdm(range(nmb_scenarios)):
                recourse_problem.solve(qualification_matrix=qualification_matrix,
                                       demand_scenario=self.demand_scenarios[scenario])
                self.add_benders_cut(scenario=scenario, recourse_problem=recourse_problem)
                lost_sales[scenario] = proba_per_scenario * recourse_problem.get_lost_sales()

            print('Solving master problem...')
            self.run()

            lb = pyo.value(self.model.objective)
            ub = self.get_qualification_costs() + float(np.sum(lost_sales))
            gap = compute_gap(lb=lb, ub=ub)
            print(f"lb: {lb}, ub: {ub}, gap: {gap} (%)")

        print('Solving process done')
