from unittest import TestCase

import numpy as np

from lib.Dataset import Dataset
from lib.DemandScenario import DemandScenario
from lib.MasterProblem import MasterProblem


class MyTestCase(TestCase):

    def test_simple_problem(self):
        nmb_products = 1
        nmb_factories = 5
        qualification_matrix = np.array([[1, 1, 1, 1, 1]], dtype=np.float64)
        qualification_costs = np.array([[2, 2, 0.5, 2, 2]], dtype=np.float64)
        lost_sales_cost = np.array([1000], dtype=np.float64)
        factory_capacities = np.array([50, 50, 50, 50, 50], dtype=np.float64)
        dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs, lost_sales_cost, factory_capacities)
        demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([10, 10, 10, 10, 10], dtype=np.float64))

        master_problem: MasterProblem = MasterProblem(dataset)
        master_problem.add_scenario(demand_scenario=demand_scenario)

        master_problem.solve()

        self.assertAlmostEqual(0.0, master_problem.get_lost_sales(scenario=0))
        self.assertAlmostEqual(0.5, master_problem.get_qualification_costs())
        self.assertAlmostEqual(0.0, master_problem.get_qualification_decision(product=0, factory=0))
        self.assertAlmostEqual(0.0, master_problem.get_qualification_decision(product=0, factory=1))
        self.assertAlmostEqual(1.0, master_problem.get_qualification_decision(product=0, factory=2))
        self.assertAlmostEqual(0.0, master_problem.get_qualification_decision(product=0, factory=3))
        self.assertAlmostEqual(0.0, master_problem.get_qualification_decision(product=0, factory=4))
