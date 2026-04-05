import math


def compute_gap(lb: float, ub: float) -> float:
    if abs(lb) == math.inf or abs(ub) == math.inf:
        return 100
    return abs(ub - lb) / (lb + 1E-10) * 100
