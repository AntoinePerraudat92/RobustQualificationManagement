from typing import Tuple

import numpy as np

from src.data_model.Dataset import Dataset
from src.data_model.DemandScenario import DemandScenario, generate_random_demand_scenario
from src.data_model.DemandUncertaintySet import DemandUncertaintySet


def generate_instance(seed: int) -> Tuple[Dataset, list[DemandScenario]]:
    # Problem dimension.
    nmb_scenarios = 50
    nmb_products = 30
    nmb_factories = 5

    # Generate demand.
    local_rng = np.random.default_rng(seed=seed)
    demand_lower_bounds = local_rng.uniform(0.0, 1000.0, size=(nmb_products,))
    demand_upper_bounds = demand_lower_bounds + local_rng.uniform(10.0, 1000.0, size=(nmb_products,))
    maximum_total_demand = np.sum(0.5 * (demand_lower_bounds + demand_upper_bounds))
    uncertainty_set: DemandUncertaintySet = DemandUncertaintySet(demand_lower_bounds=demand_lower_bounds,
                                                                 demand_upper_bounds=demand_upper_bounds,
                                                                 maximum_total_demand=maximum_total_demand)
    demand_scenarios: list[DemandScenario] = [
        generate_random_demand_scenario(nmb_products=nmb_products, seed=scenario,
                                        uncertainty_set=uncertainty_set) for scenario in range(nmb_scenarios)]

    # Generate dataset.
    qualification_matrix = np.array(local_rng.uniform(0.0, 1.0, size=(nmb_products, nmb_factories,)) <= 0.50,
                                    dtype=np.float64)
    qualification_costs = local_rng.uniform(1.0, 50.0, size=(nmb_products, nmb_factories,))
    lost_sales_cost = local_rng.uniform(1.0, 10000.0, size=(nmb_products,))
    total_demand_per_factory = maximum_total_demand / nmb_factories
    factory_capacities = local_rng.uniform(total_demand_per_factory * 0.80, total_demand_per_factory * 1.20,
                                           size=(nmb_factories,))
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs, lost_sales_cost,
                               factory_capacities)

    return dataset, demand_scenarios
