from lib.Dataset import Dataset
from lib.DemandScenario import DemandScenario, generate_random_demand_scenario
from lib.DemandUncertaintySet import DemandUncertaintySet
from lib.MasterProblem import MasterProblem
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from lib.RecourseProblem import RecourseProblem


def compute_gap(lb: float, ub: float) -> float:
    return abs(ub - lb) / (lb + 1E-10)


class CCGSolver:

    def __init__(self, dataset: Dataset, nmb_scenarios: int, uncertainty_set: DemandUncertaintySet):
        self.dataset = dataset
        self.master_problem: MasterProblem = MasterProblem(self.dataset)
        self.demand_scenarios: list[DemandScenario] = [
            generate_random_demand_scenario(nmb_products=self.dataset.nmb_products, seed=scenario,
                                            uncertainty_set=uncertainty_set) for scenario in range(nmb_scenarios)]

    def solve(self):
        lb: float = 0.0
        ub: float = 1.0
        gap = compute_gap(lb=lb, ub=ub)
        print(f"lb: {lb}, ub: {ub}, gap: {gap}")
        nmb_scenarios = len(self.demand_scenarios)
        # We need an initial solution to get the qualification matrix. We use the first scenario by default as
        # the baseline.
        self.master_problem.add_scenario(demand_scenario=self.demand_scenarios[0])
        self.master_problem.solve()
        while gap > 1E-4:
            qualification_matrix: NDArray[np.int64] = self.master_problem.get_qualification_matrix()
            lost_sales: NDArray[np.float64] = np.zeros(shape=nmb_scenarios, dtype=np.float64)

            print('Solving recourse problems...')
            for scenario in tqdm(range(nmb_scenarios)):
                recourse_problem: RecourseProblem = RecourseProblem(self.dataset)
                recourse_problem.build(qualification_matrix=qualification_matrix,
                                       demand_scenario=self.demand_scenarios[scenario])
                recourse_problem.solve()
                lost_sales[scenario] = recourse_problem.get_lost_sales()

            worst_scenario = int(np.argmax(lost_sales))
            worst_lost_sales = float(lost_sales[worst_scenario])
            print(f'Worst scenario: {worst_scenario}, worst lost sales: {worst_lost_sales}')

            print('Solving master problem...')
            self.master_problem.add_scenario(demand_scenario=self.demand_scenarios[worst_scenario])
            self.master_problem.solve()

            lb = self.master_problem.get_objective_function()
            ub = self.master_problem.get_qualification_costs() + worst_lost_sales
            gap = compute_gap(lb=lb, ub=ub)
            print(f"lb: {lb}, ub: {ub}, gap: {gap}")

        print('Solving process done')

    def get_qualification_costs(self) -> float:
        return self.master_problem.get_qualification_costs()
