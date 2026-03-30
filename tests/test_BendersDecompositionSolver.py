import pytest
import numpy as np

from src.data_model.Dataset import Dataset
from src.data_model.DemandScenario import DemandScenario
from src.solver.stochastic.BendersDecompositionSolver import BendersDecompositionSolver


def test_simple_problem_with_one_expensive_qualification():
    nmb_products = 1
    nmb_factories = 1
    qualification_matrix = np.array([[1]], dtype=np.float64)
    qualification_costs = np.array([[10000]], dtype=np.float64)
    lost_sales_cost = np.array([1], dtype=np.float64)
    factory_capacities = np.array([20], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                               lost_sales_cost, factory_capacities)
    first_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([10], dtype=np.float64))
    second_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([25], dtype=np.float64))
    demand_scenarios = [first_demand_scenario, second_demand_scenario]

    solver: BendersDecompositionSolver = BendersDecompositionSolver(dataset=dataset, demand_scenarios=demand_scenarios)
    solver.solve()

    assert 0.0 == pytest.approx(solver.get_qualification_costs())
    assert 17.5 == pytest.approx(solver.get_lost_sales())


def test_simple_problem_with_one_product_and_one_factory():
    nmb_products = 1
    nmb_factories = 1
    qualification_matrix = np.array([[1]], dtype=np.float64)
    qualification_costs = np.array([[0.1]], dtype=np.float64)
    lost_sales_cost = np.array([1], dtype=np.float64)
    factory_capacities = np.array([20], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                               lost_sales_cost, factory_capacities)
    first_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([10], dtype=np.float64))
    second_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([25], dtype=np.float64))
    demand_scenarios = [first_demand_scenario, second_demand_scenario]

    solver: BendersDecompositionSolver = BendersDecompositionSolver(dataset=dataset, demand_scenarios=demand_scenarios)
    solver.solve()

    assert 0.1 == pytest.approx(solver.get_qualification_costs())
    assert 2.5 == pytest.approx(solver.get_lost_sales())


def test_simple_problem_with_one_product_and_three_factories():
    nmb_products = 1
    nmb_factories = 3
    qualification_matrix = np.array([[1, 1, 1]], dtype=np.float64)
    qualification_costs = np.array([[1, 2, 5]], dtype=np.float64)
    lost_sales_cost = np.array([10], dtype=np.float64)
    factory_capacities = np.array([10, 5, 10], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                               lost_sales_cost, factory_capacities)
    demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([25], dtype=np.float64))
    demand_scenarios = [demand_scenario]

    solver: BendersDecompositionSolver = BendersDecompositionSolver(dataset=dataset, demand_scenarios=demand_scenarios)
    solver.solve()

    assert 8.0 == pytest.approx(solver.get_qualification_costs())
    assert 0.0 == pytest.approx(solver.get_lost_sales())


def test_simple_problem_with_two_scenarios_and_dedicated_factories():
    nmb_products = 2
    nmb_factories = 2
    qualification_matrix = np.array([[1, 0], [0, 1]], dtype=np.float64)
    qualification_costs = np.array([[1, 0], [0, 1]], dtype=np.float64)
    lost_sales_cost = np.array([50, 50], dtype=np.float64)
    factory_capacities = np.array([15, 15], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs, lost_sales_cost,
                               factory_capacities)
    first_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([15, 0], dtype=np.float64))
    second_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([0, 15], dtype=np.float64))
    demand_scenarios = [first_demand_scenario, second_demand_scenario]

    solver: BendersDecompositionSolver = BendersDecompositionSolver(dataset=dataset, demand_scenarios=demand_scenarios)
    solver.solve()

    assert 2.0 == pytest.approx(solver.get_qualification_costs())
    assert 0.0 == pytest.approx(solver.get_lost_sales())
    assert 1.0 == pytest.approx(solver.get_qualification_decision(product=0, factory=0))
    assert 0.0 == pytest.approx(solver.get_qualification_decision(product=0, factory=1))
    assert 0.0 == pytest.approx(solver.get_qualification_decision(product=1, factory=0))
    assert 1.0 == pytest.approx(solver.get_qualification_decision(product=1, factory=1))
