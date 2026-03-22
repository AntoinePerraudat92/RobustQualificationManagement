import numpy as np

from lib.Dataset import Dataset
from lib.DemandScenario import DemandScenario
from lib.DemandUncertaintySet import DemandUncertaintySet


def generate_random_demand_scenario(nmb_products: int, seed: int, uncertainty_set: DemandUncertaintySet) -> DemandScenario:
    local_random = np.random.default_rng(seed)
    product_demands = [
        local_random.uniform(uncertainty_set.demand_lower_bounds[p], uncertainty_set.demand_upper_bounds[p]) for p
        in range(nmb_products)]
    product_demands = np.array(product_demands, dtype=np.float64)
    total_demand = np.sum(product_demands)
    return DemandScenario(product_demands=uncertainty_set.maximum_total_demand / total_demand * product_demands)


def generate_random_dataset(nmb_products: int, nmb_factories: int, seed: int) -> Dataset:
    local_random = np.random.default_rng(seed)





