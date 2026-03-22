import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass


@dataclass
class DemandScenario:
    product_demands: NDArray[np.float64]
