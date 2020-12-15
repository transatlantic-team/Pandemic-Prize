import numpy as np

def mae(pred, true):
    """Compute the Mean Absolute Error.

    >>> from transat.metric import mae
    """
    return np.mean(np.abs(np.array(pred) - np.array(true)))