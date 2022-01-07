import numpy as np


def relu(Z: np.array) -> tuple:
    """
    Implements forward propagation for a single ReLU unit.
    :param Z: numpy array of any shape
    """
    A = np.maximum(0, Z)
    return A


def softmax(Z: np.array) -> tuple:
    """
    Implements forward propagation for a single Softmax unit.
    :param Z: numpy array of any shape
    :return A: post-activation parameter A, of the same shape as z
    """
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def relu_backward(Z: np.array) -> np.array:
    """
    Implements backward propagation for a single ReLU unit.
    :param dA: post-activation gradient dA, of any shape
    :return dZ: gradient of the cost with respect to Z
    """
    dZ = Z > 0
    return dZ
