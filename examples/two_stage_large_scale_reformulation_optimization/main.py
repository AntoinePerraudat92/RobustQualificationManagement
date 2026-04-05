from examples.dataset.InstanceGenerator import generate_instance
from src.solver.stochastic.LargeScaleReformulationSolver import LargeScaleReformulationSolver


def main(seed: int):
    dataset, demand_scenarios = generate_instance(seed=seed)
    solver: LargeScaleReformulationSolver = LargeScaleReformulationSolver(dataset=dataset,
                                                                          demand_scenarios=demand_scenarios,
                                                                          w=0.2,
                                                                          alpha=0.90)
    solver.solve()
    qualification_costs = solver.get_qualification_costs()
    lost_sales = solver.get_expected_lost_sales()
    cvar = solver.get_cvar()
    print(f"Qualification costs: {qualification_costs}")
    print(f"Lost sales: {lost_sales}")
    print(f"CVaR: {cvar}")


if __name__ == '__main__':
    main(seed=1234)
