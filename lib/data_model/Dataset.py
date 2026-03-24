import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Dataset:
    nmb_products: int
    nmb_factories: int
    qualification_matrix: NDArray[np.int64]
    qualification_costs: NDArray[np.float64]
    lost_sales_costs: NDArray[np.float64]
    factory_capacities: NDArray[np.float64]
