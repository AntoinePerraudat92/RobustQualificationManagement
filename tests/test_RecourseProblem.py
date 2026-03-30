import pytest
import numpy as np

from src.data_model.Dataset import Dataset
from src.data_model.DemandScenario import DemandScenario
from src.solver.robust.RecourseProblem import RecourseProblem
from src.solver.stochastic.DualRecourseProblem import DualRecourseProblem


@pytest.mark.parametrize("cls", [RecourseProblem, DualRecourseProblem])
def test_simple_problem_with_one_product_and_one_factory(cls):
    nmb_products = 1
    nmb_factories = 1
    qualification_matrix = np.array([[1]], dtype=np.float64)
    qualification_costs = np.array([[2]], dtype=np.float64)
    lost_sales_cost = np.array([10], dtype=np.float64)
    factory_capacities = np.array([2], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                               lost_sales_cost, factory_capacities)
    demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([4], dtype=np.float64))

    problem = cls(dataset)

    feasible = problem.solve(qualification_matrix=qualification_matrix, demand_scenario=demand_scenario)
    assert feasible is True
    assert 20.0 == pytest.approx(problem.get_lost_sales())


@pytest.mark.parametrize("cls", [RecourseProblem, DualRecourseProblem])
def test_simple_problem_with_one_product_and_no_factory(cls):
    nmb_products = 1
    nmb_factories = 0
    qualification_matrix = np.array([[0]], dtype=np.float64)
    qualification_costs = np.array([[2]], dtype=np.float64)
    lost_sales_cost = np.array([10], dtype=np.float64)
    factory_capacities = np.array([2], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                               lost_sales_cost, factory_capacities)
    demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([4], dtype=np.float64))

    problem = cls(dataset)

    feasible = problem.solve(qualification_matrix=qualification_matrix, demand_scenario=demand_scenario)
    assert feasible is True
    assert 40.0 == pytest.approx(problem.get_lost_sales())


@pytest.mark.parametrize("cls", [RecourseProblem, DualRecourseProblem])
def test_simple_problem_with_two_product_and_one_factory(cls):
    nmb_products = 2
    nmb_factories = 1
    qualification_matrix = np.array([[1], [1]], dtype=np.float64)
    qualification_costs = np.array([[1], [1]], dtype=np.float64)
    lost_sales_cost = np.array([10, 30], dtype=np.float64)
    factory_capacities = np.array([25], dtype=np.float64)
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs,
                               lost_sales_cost, factory_capacities)
    demand_scenario: DemandScenario = DemandScenario(product_demands=np.array([5, 40], dtype=np.float64))

    problem = cls(dataset)

    feasible = problem.solve(qualification_matrix=qualification_matrix, demand_scenario=demand_scenario)
    assert feasible is True
    assert 500.0 == pytest.approx(problem.get_lost_sales())

