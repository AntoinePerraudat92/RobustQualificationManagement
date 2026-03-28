from unittest import TestCase

import numpy as np

from src.data_model.Dataset import Dataset
from src.data_model.DemandScenario import DemandScenario
from src.solver.CCGSolver import CCGSolver


class CCGSolverTest(TestCase):

    def test_simple_problem_with_two_products_and_two_factories(self):
        nmb_products = 2
        nmb_factories = 2
        qualification_matrix = np.array([[1, 0], [0, 1]], dtype=np.float64)
        qualification_costs = np.array([[1, 10], [10, 1]], dtype=np.float64)
        lost_sales_cost = np.array([500, 500], dtype=np.float64)
        factory_capacities = np.array([100, 100], dtype=np.float64)
        dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                                   lost_sales_cost, factory_capacities)
        first_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([10, 0], dtype=np.float64))
        second_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([0, 10], dtype=np.float64))

        solver: CCGSolver = CCGSolver(dataset)
        solver.solve(demand_scenarios=[first_demand_scenario, second_demand_scenario])

        self.assertAlmostEqual(2.0, solver.get_qualification_costs())
