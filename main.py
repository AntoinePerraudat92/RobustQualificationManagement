import numpy as np

from lib.solver.CCGSolver import CCGSolver
from lib.data_model.Dataset import Dataset
from lib.data_model.DemandScenario import DemandScenario, generate_random_demand_scenario
from lib.data_model.DemandUncertaintySet import DemandUncertaintySet


def main(seed: int):
    # Problem dimension.
    nmb_scenarios = 100
    nmb_products = 300
    nmb_factories = 12

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
                                    dtype=np.int64)
    qualification_costs = local_rng.uniform(1.0, 50.0, size=(nmb_products, nmb_factories,))
    lost_sales_cost = local_rng.uniform(10.0, 1000.0, size=(nmb_products,))
    total_demand_per_factory = (maximum_total_demand * nmb_products) / nmb_factories
    factory_capacities = local_rng.uniform(total_demand_per_factory * 0.90, total_demand_per_factory, size=(nmb_factories,))
    dataset: Dataset = Dataset(nmb_products, nmb_factories, qualification_matrix, qualification_costs, lost_sales_cost,
                               factory_capacities)

    # Solve two-stage robust optimization problem.
    solver: CCGSolver = CCGSolver(dataset)
    solver.solve(demand_scenarios=demand_scenarios)
    qualification_costs = solver.get_qualification_costs()
    print(f"Qualification costs: {qualification_costs}")


if __name__ == '__main__':
    main(seed=1234)
