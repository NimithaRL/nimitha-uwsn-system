import numpy as np
def compute_deviation(current, previous):
    return float(np.mean(np.abs(current - previous)))
