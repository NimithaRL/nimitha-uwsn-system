import numpy as np
def aggregate(models):
    return (np.mean([m.coef_ for m in models], axis=0),
            np.mean([m.intercept_ for m in models], axis=0))
