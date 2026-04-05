import pytest
import numpy as np

from src.data_model.Dataset import Dataset
from src.data_model.DemandScenario import DemandScenario
from src.solver.stochastic.LargeScaleReformulationSolver import LargeScaleReformulationSolver


def test_simple_problem_with_one_qualification():
    nmb_products = 1
    nmb_factories = 1
    qualification_matrix = np.array([[1]], dtype=np.float64)
    qualification_costs = np.array([[1.5]], dtype=np.float64)
    lost_sales_cost = np.array([1], dtype=np.float64)
    factory_capacities = np.array([20], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                               lost_sales_cost, factory_capacities)
    demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([10], dtype=np.float64))
    demand_scenarios = [demand_scenario]

    solver: LargeScaleReformulationSolver = LargeScaleReformulationSolver(dataset=dataset,
                                                                          demand_scenarios=demand_scenarios)
    solver.solve()

    assert 1.5 == pytest.approx(solver.get_qualification_costs())
    assert 0.0 == pytest.approx(solver.get_expected_lost_sales())
    assert 0.0 == pytest.approx(solver.get_cvar())


@pytest.mark.parametrize("w", [0.0, 0.25, 0.50, 0.75, 1.0])
@pytest.mark.parametrize("alpha", [0.01, 0.10, 0.90, 0.99])
def test_simple_problem_with_lost_sales(w, alpha):
    nmb_products = 1
    nmb_factories = 1
    qualification_matrix = np.array([[1]], dtype=np.float64)
    qualification_costs = np.array([[1]], dtype=np.float64)
    lost_sales_cost = np.array([5], dtype=np.float64)
    factory_capacities = np.array([20], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                               lost_sales_cost, factory_capacities)
    demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([100], dtype=np.float64))
    demand_scenarios = [demand_scenario]

    solver: LargeScaleReformulationSolver = LargeScaleReformulationSolver(dataset=dataset,
                                                                          demand_scenarios=demand_scenarios,
                                                                          w=w,
                                                                          alpha=alpha)
    solver.solve()

    assert 1.0 == pytest.approx(solver.get_qualification_costs())
    assert 400.0 == pytest.approx(solver.get_expected_lost_sales())
    assert 400.0 == pytest.approx(solver.get_cvar())


def test_simple_problem_with_three_scenarios():
    nmb_products = 1
    nmb_factories = 1
    qualification_matrix = np.array([[1]], dtype=np.float64)
    qualification_costs = np.array([[1]], dtype=np.float64)
    lost_sales_cost = np.array([1], dtype=np.float64)
    factory_capacities = np.array([20], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                               lost_sales_cost, factory_capacities)
    first_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([10], dtype=np.float64))
    second_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([25], dtype=np.float64))
    third_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([100], dtype=np.float64))
    demand_scenarios = [first_demand_scenario, second_demand_scenario, third_demand_scenario]

    solver: LargeScaleReformulationSolver = LargeScaleReformulationSolver(dataset=dataset,
                                                                          demand_scenarios=demand_scenarios,
                                                                          w=0.75,
                                                                          alpha=0.95)
    solver.solve()

    assert 1.0 == pytest.approx(solver.get_qualification_costs())
    assert 68.08333 == pytest.approx(solver.get_objective_function())
    assert 28.33333 == pytest.approx(solver.get_expected_lost_sales())
    assert 80.0 == pytest.approx(solver.get_cvar())


@pytest.mark.parametrize("w, alpha, expected_objective_function",
                         [(0.0, 0.0, 42.0), (1.0, 0.0, 42.0), (1.0, 0.95, 142.0), (0.50, 0.50, 62.0)])
def test_simple_problem_with_two_products_dedicated_factories(w, alpha, expected_objective_function):
    nmb_products = 2
    nmb_factories = 2
    qualification_matrix = np.array([[1, 0], [0, 1]], dtype=np.float64)
    qualification_costs = np.array([[1, 0], [0, 1]], dtype=np.float64)
    lost_sales_cost = np.array([1, 1], dtype=np.float64)
    factory_capacities = np.array([30, 30], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs, lost_sales_cost,
                               factory_capacities)
    first_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([25, 25], dtype=np.float64))
    second_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([30, 30], dtype=np.float64))
    third_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([40, 40], dtype=np.float64))
    fourth_demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([100, 100], dtype=np.float64))
    demand_scenarios = [first_demand_scenario, second_demand_scenario, third_demand_scenario, fourth_demand_scenario]

    solver: LargeScaleReformulationSolver = LargeScaleReformulationSolver(dataset=dataset,
                                                                          demand_scenarios=demand_scenarios,
                                                                          w=w,
                                                                          alpha=alpha)

    solver.solve()

    assert 2.0 == pytest.approx(solver.get_qualification_costs())
    assert expected_objective_function == pytest.approx(solver.get_objective_function())
