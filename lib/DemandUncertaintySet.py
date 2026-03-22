import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class DemandUncertaintySet:
    demand_lower_bounds: NDArray[np.float64]
    demand_upper_bounds: NDArray[np.float64]
    maximum_total_demand: np.float64
