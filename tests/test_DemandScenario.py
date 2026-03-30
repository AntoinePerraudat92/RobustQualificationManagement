import pytest
import numpy as np

from src.data_model.DemandScenario import generate_random_demand_scenario
from src.data_model.DemandUncertaintySet import DemandUncertaintySet


def test_generate_random_demand_scenario_for_no_product():
    nmb_products = 0
    seed = 0
    uncertainty_set: DemandUncertaintySet = DemandUncertaintySet(
        demand_lower_bounds=np.empty(shape=()),
        demand_upper_bounds=np.empty(shape=()),
        maximum_total_demand=np.float64(0.0)
    )

    with pytest.raises(RuntimeError) as excinfo:
        generate_random_demand_scenario(nmb_products, seed, uncertainty_set)
    assert 'Number of products must be larger than 0' in str(excinfo.value)


def test_generate_random_demand_scenario_for_two_products():
    nmb_products = 2
    seed = 0
    uncertainty_set: DemandUncertaintySet = DemandUncertaintySet(
        demand_lower_bounds=np.array([0, 0], dtype=np.float64),
        demand_upper_bounds=np.array([0, 0], dtype=np.float64),
        maximum_total_demand=np.float64(10.0)
    )

    demand_scenario = generate_random_demand_scenario(nmb_products, seed, uncertainty_set)

    assert uncertainty_set.maximum_total_demand == pytest.approx(np.sum(demand_scenario.product_demands))


def test_generate_random_demand_scenario_for_three_products():
    nmb_products = 3
    seed = 0
    uncertainty_set: DemandUncertaintySet = DemandUncertaintySet(
        demand_lower_bounds=np.array([0, 15, 10], dtype=np.float64),
        demand_upper_bounds=np.array([0, 25, 40], dtype=np.float64),
        maximum_total_demand=np.float64(40.0)
    )

    demand_scenario = generate_random_demand_scenario(nmb_products, seed, uncertainty_set)

    assert uncertainty_set.maximum_total_demand == pytest.approx(np.sum(demand_scenario.product_demands))
