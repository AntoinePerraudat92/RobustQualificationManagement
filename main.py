import numpy as np

from lib.CCGSolver import CCGSolver
from lib.Dataset import Dataset
from lib.DemandUncertaintySet import DemandUncertaintySet


def main(seed: int):
    local_rng = np.random.default_rng(seed=seed)
    nmb_products = 70
    nmb_factories = 10

    # Generate demand.
    demand_lower_bounds = local_rng.uniform(300.0, 1000.0, size=(nmb_products,))
    demand_upper_bounds = demand_lower_bounds + local_rng.uniform(300.0, 1000.0, size=(nmb_products,))
    maximum_total_demand = np.sum(0.5 * (demand_lower_bounds + demand_upper_bounds))
    uncertainty_set: DemandUncertaintySet = DemandUncertaintySet(demand_lower_bounds=demand_lower_bounds,
                                                                 demand_upper_bounds=demand_upper_bounds,
                                                                 maximum_total_demand=maximum_total_demand)

    # Generate dataset.
    qualification_matrix = np.array(local_rng.uniform(0.0, 1.0, size=(nmb_products, nmb_factories,)) <= 0.50,
                                    dtype=np.int64)
    qualification_costs = local_rng.uniform(1.0, 50.0, size=(nmb_products, nmb_factories,))
    lost_sales_cost = local_rng.uniform(10.0, 1000.0, size=(nmb_products,))
    total_demand_per_factory = (maximum_total_demand * nmb_products) / nmb_factories
    factory_capacities = local_rng.uniform(total_demand_per_factory * 0.90, total_demand_per_factory, size=(nmb_factories,))
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs, lost_sales_cost,
                               factory_capacities)

    solver: CCGSolver = CCGSolver(dataset, 500, uncertainty_set)
    solver.solve()


if __name__ == '__main__':
    main(seed=1234)
