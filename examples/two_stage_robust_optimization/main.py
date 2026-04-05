from examples.dataset.InstanceGenerator import generate_instance
from src.solver.robust.CCGSolver import CCGSolver


def main(seed: int):
    dataset, demand_scenarios = generate_instance(seed=seed)
    solver: CCGSolver = CCGSolver(dataset)
    solver.solve(demand_scenarios=demand_scenarios)
    qualification_costs = solver.get_qualification_costs()
    lost_sales = solver.get_lost_sales()
    print(f"Qualification costs: {qualification_costs}")
    print(f"Lost sales: {lost_sales}")


if __name__ == '__main__':
    main(seed=1234)
