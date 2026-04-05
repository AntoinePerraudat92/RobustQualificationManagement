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


@pytest.mark.parametrize("w", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("alpha", [0.01, 0.10, 0.90, 0.99])
def test_simple_problem_with_lost_sales(w, alpha):
    nmb_products = 1
    nmb_factories = 1
    qualification_matrix = np.array([[1]], dtype=np.float64)
    qualification_costs = np.array([[1]], dtype=np.float64)
    lost_sales_cost = np.array([1], dtype=np.float64)
    factory_capacities = np.array([20], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                               lost_sales_cost, factory_capacities)
    demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([100], dtype=np.float64))
    demand_scenarios = [demand_scenario]

    solver: LargeScaleReformulationSolver = LargeScaleReformulationSolver(dataset=dataset,
                                                                          demand_scenarios=demand_scenarios)
    solver.solve()

    assert 1.0 == pytest.approx(solver.get_qualification_costs())
    assert 40.0 == pytest.approx(solver.get_expected_lost_sales())
    assert 40.0 == pytest.approx(solver.get_cvar())
