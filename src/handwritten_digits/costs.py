import numpy as np


def compute_cross_entropy_cost(AL: np.array, Y: np.array) -> float:
    """
    Implements the cross-entropy cost function.
    :param AL: probability vector with label predictions
    :param Y: true label vector
    :return: cross-entropy cost
    """
    m = Y.shape[1]
    cost = -1 / m * np.sum((Y * np.log(AL) + (1 - Y) * np.log(1 - AL)))
    cost = np.squeeze(cost)  # this turns e.g. [[17]] into 17
    return cost
