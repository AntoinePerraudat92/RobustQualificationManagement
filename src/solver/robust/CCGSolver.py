from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import math

from src.solver.robust.RecourseProblem import RecourseProblem
from src.data_model.Dataset import Dataset
from src.data_model.DemandScenario import DemandScenario
from src.solver.robust.MasterProblem import MasterProblem


def compute_gap(lb: float, ub: float) -> float:
    if abs(lb) == math.inf or abs(ub) == math.inf:
        return 100
    return abs(ub - lb) / (lb + 1E-10) * 100


class CCGSolver:

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.master_problem: MasterProblem = MasterProblem(self.dataset)

    def solve(self, demand_scenarios: list[DemandScenario]):
        recourse_problem: RecourseProblem = RecourseProblem(self.dataset)
        lb: float = -math.inf
        ub: float = math.inf
        gap = compute_gap(lb=lb, ub=ub)
        print(f"lb: {lb}, ub: {ub}, gap: {gap} (%)")
        nmb_scenarios = len(demand_scenarios)
        # We solve the master problem for any scenario to have meaningful qualification decisions for the recourse problem.
        self.master_problem.add_scenario(demand_scenario=demand_scenarios[0])
        self.master_problem.solve()
        while gap > 1E-4:
            qualification_matrix: NDArray[np.int64] = self.master_problem.get_qualification_matrix()
            lost_sales: NDArray[np.float64] = np.zeros(shape=(nmb_scenarios,), dtype=np.float64)

            print('Solving recourse problems...')
            for scenario in tqdm(range(nmb_scenarios)):
                recourse_problem.solve(qualification_matrix=qualification_matrix,
                                       demand_scenario=demand_scenarios[scenario])
                lost_sales[scenario] = recourse_problem.get_lost_sales()

            worst_scenario = int(np.argmax(lost_sales))
            worst_lost_sales = float(lost_sales[worst_scenario])
            print(f'Worst scenario: {worst_scenario}, worst lost sales: {worst_lost_sales}')

            print('Solving master problem...')
            self.master_problem.add_scenario(demand_scenario=demand_scenarios[worst_scenario])
            self.master_problem.solve()

            lb = self.master_problem.get_objective_function()
            ub = self.master_problem.get_qualification_costs() + worst_lost_sales
            gap = compute_gap(lb=lb, ub=ub)
            print(f"lb: {lb}, ub: {ub}, gap: {gap} (%)")

        print('Solving process done')

    def get_qualification_costs(self) -> float:
        return self.master_problem.get_qualification_costs()

    def get_lost_sales(self) -> float:
        return self.master_problem.get_worst_case_lost_sales()
