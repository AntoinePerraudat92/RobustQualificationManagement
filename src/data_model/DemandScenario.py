import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass

from src.data_model.DemandUncertaintySet import DemandUncertaintySet


@dataclass(frozen=True, eq=True)
class DemandScenario:
    product_demands: NDArray[np.float64]


def generate_random_demand_scenario(nmb_products: int, seed: int, uncertainty_set: DemandUncertaintySet) -> DemandScenario:
    local_random = np.random.default_rng(seed)
    product_demands = [
        local_random.uniform(uncertainty_set.demand_lower_bounds[p], uncertainty_set.demand_upper_bounds[p]) for p
        in range(nmb_products)]
    product_demands = np.array(product_demands, dtype=np.float64)
    total_demand = np.sum(product_demands)
    return DemandScenario(product_demands=uncertainty_set.maximum_total_demand / total_demand * product_demands)


