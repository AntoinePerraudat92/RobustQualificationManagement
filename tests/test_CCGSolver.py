import pytest
import numpy as np

from src.data_model.Dataset import Dataset
from src.data_model.DemandScenario import DemandScenario
from src.solver.robust.CCGSolver import CCGSolver


def test_simple_problem_with_two_products_and_two_factories():
    nmb_products = 2
    nmb_factories = 2
    qualification_matrix = np.array([[1, 0], [0, 1]], dtype=np.float64)
    qualification_costs = np.array([[4.2, 10], [10, 2]], dtype=np.float64)
    lost_sales_cost = np.array([500, 500], dtype=np.float64)
    factory_capacities = np.array([100, 100], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                               lost_sales_cost, factory_capacities)
    first_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([10, 0], dtype=np.float64))
    second_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([0, 10], dtype=np.float64))

    solver: CCGSolver = CCGSolver(dataset)
    solver.solve(demand_scenarios=[first_demand_scenario, second_demand_scenario])

    assert 6.2 == pytest.approx(solver.get_qualification_costs())
    assert 0.0 == pytest.approx(solver.get_lost_sales())


def test_simple_problem_with_one_product_and_lost_sales():
    nmb_products = 1
    nmb_factories = 2
    qualification_matrix = np.array([[1, 1]], dtype=np.float64)
    qualification_costs = np.array([[1, 1]], dtype=np.float64)
    lost_sales_cost = np.array([10], dtype=np.float64)
    factory_capacities = np.array([25, 25], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                               lost_sales_cost, factory_capacities)
    first_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([5], dtype=np.float64))
    second_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([60], dtype=np.float64))

    solver: CCGSolver = CCGSolver(dataset)
    solver.solve(demand_scenarios=[first_demand_scenario, second_demand_scenario])

    assert 2.0 == pytest.approx(solver.get_qualification_costs())
    assert 100.0 == pytest.approx(solver.get_lost_sales())
