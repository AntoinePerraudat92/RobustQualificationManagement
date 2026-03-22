from unittest import TestCase

import numpy as np

from lib.DemandUncertaintySet import DemandUncertaintySet
from lib.RandomDataGenerator import generate


class Test(TestCase):

    def generate_random_demand_scenario(self):
        nmb_products = 3
        seed = 0
        uncertainty_set: DemandUncertaintySet = DemandUncertaintySet(
            demand_lower_bounds=np.array([0, 15, 10], dtype=np.float64),
            demand_upper_bounds=np.array([0, 25, 40], dtype=np.float64),
            maximum_total_demand=np.float64(40.0)
        )

        demand_scenario = generate(nmb_products, seed, uncertainty_set)

        self.assertAlmostEqual(uncertainty_set.maximum_total_demand, np.sum(demand_scenario.product_demands))

    def generate_random_dataset(self):
        nmb_products = 3
        seed = 0
        uncertainty_set: DemandUncertaintySet = DemandUncertaintySet(
            demand_lower_bounds=np.array([0, 15, 10], dtype=np.float64),
            demand_upper_bounds=np.array([0, 25, 40], dtype=np.float64),
            maximum_total_demand=np.float64(40.0)
        )

        demand_scenario = generate(nmb_products, seed, uncertainty_set)

        self.assertAlmostEqual(uncertainty_set.maximum_total_demand, np.sum(demand_scenario.product_demands))
