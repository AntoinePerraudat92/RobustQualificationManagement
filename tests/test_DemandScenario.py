from unittest import TestCase

import numpy as np

from src.data_model.DemandScenario import generate_random_demand_scenario
from src.data_model.DemandUncertaintySet import DemandUncertaintySet


class DemandScenarioTest(TestCase):

    def assertRaisesWithMessage(self, exception_type, message, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except exception_type as e:
            self.assertEqual(e.args[0], message)
        else:
            self.fail('"{0}" was expected to throw "{1}" exception'
                      .format(func.__name__, exception_type.__name__))

    def test_generate_random_demand_scenario_for_no_product(self):
        nmb_products = 0
        seed = 0
        uncertainty_set: DemandUncertaintySet = DemandUncertaintySet(
            demand_lower_bounds=np.empty(shape=()),
            demand_upper_bounds=np.empty(shape=()),
            maximum_total_demand=np.float64(0.0)
        )

        self.assertRaisesWithMessage(RuntimeError, 'Number of products must be larger than 0',
                                     generate_random_demand_scenario, nmb_products, seed, uncertainty_set)

    def test_generate_random_demand_scenario_for_two_products(self):
        nmb_products = 2
        seed = 0
        uncertainty_set: DemandUncertaintySet = DemandUncertaintySet(
            demand_lower_bounds=np.array([0, 0], dtype=np.float64),
            demand_upper_bounds=np.array([0, 0], dtype=np.float64),
            maximum_total_demand=np.float64(10.0)
        )

        demand_scenario = generate_random_demand_scenario(nmb_products, seed, uncertainty_set)

        self.assertAlmostEqual(uncertainty_set.maximum_total_demand, np.sum(demand_scenario.product_demands))

    def test_generate_random_demand_scenario_for_three_products(self):
        nmb_products = 3
        seed = 0
        uncertainty_set: DemandUncertaintySet = DemandUncertaintySet(
            demand_lower_bounds=np.array([0, 15, 10], dtype=np.float64),
            demand_upper_bounds=np.array([0, 25, 40], dtype=np.float64),
            maximum_total_demand=np.float64(40.0)
        )

        demand_scenario = generate_random_demand_scenario(nmb_products, seed, uncertainty_set)

        self.assertAlmostEqual(uncertainty_set.maximum_total_demand, np.sum(demand_scenario.product_demands))
