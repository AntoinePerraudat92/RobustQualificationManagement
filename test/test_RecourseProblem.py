from unittest import TestCase

import numpy as np

from lib.data_model.Dataset import Dataset
from lib.data_model.DemandScenario import DemandScenario
from lib.solver.RecourseProblem import RecourseProblem


class Test(TestCase):

    def test_simple_problem_with_one_product_and_one_factory(self):
        nmb_products = 1
        nmb_factories = 1
        qualification_matrix = np.array([[1]], dtype=np.float64)
        qualification_costs = np.array([[2]], dtype=np.float64)
        lost_sales_cost = np.array([10], dtype=np.float64)
        factory_capacities = np.array([2], dtype=np.float64)
        dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                                   lost_sales_cost, factory_capacities)
        demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([4], dtype=np.float64))

        recourse_problem: RecourseProblem = RecourseProblem(dataset)
        recourse_problem.build(qualification_matrix, demand_scenario)
        recourse_problem.solve()

        self.assertAlmostEqual(20.0, recourse_problem.get_lost_sales())

    def test_simple_problem_with_one_product_and_no_factory(self):
        nmb_products = 1
        nmb_factories = 1
        qualification_matrix = np.array([[0]], dtype=np.float64)
        qualification_costs = np.array([[2]], dtype=np.float64)
        lost_sales_cost = np.array([10], dtype=np.float64)
        factory_capacities = np.array([2], dtype=np.float64)
        dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                                   lost_sales_cost, factory_capacities)
        demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([4], dtype=np.float64))

        recourse_problem: RecourseProblem = RecourseProblem(dataset)
        recourse_problem.build(qualification_matrix, demand_scenario)
        recourse_problem.solve()

        self.assertAlmostEqual(40.0, recourse_problem.get_lost_sales())

    def test_simple_problem_with_two_product_and_one_factory(self):
        nmb_products = 2
        nmb_factories = 1
        qualification_matrix = np.array([[1], [1]], dtype=np.float64)
        qualification_costs = np.array([[1], [1]], dtype=np.float64)
        lost_sales_cost = np.array([10, 30], dtype=np.float64)
        factory_capacities = np.array([25], dtype=np.float64)
        dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                                   lost_sales_cost, factory_capacities)
        demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([5, 40], dtype=np.float64))

        recourse_problem: RecourseProblem = RecourseProblem(dataset)
        recourse_problem.build(qualification_matrix, demand_scenario)
        recourse_problem.solve()

        self.assertAlmostEqual(500.0, recourse_problem.get_lost_sales())
