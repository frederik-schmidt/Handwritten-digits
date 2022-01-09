import numpy as np


def relu(Z: np.array) -> np.array:
    """
    Implements forward propagation for a single ReLU unit.
    :param Z: numpy array of any shape
    :return A: post-activation parameter A, of the same shape as Z
    """
    A = np.maximum(0, Z)
    return A


def softmax(Z: np.array) -> np.array:
    """
    Implements forward propagation for a single Softmax unit.
    :param Z: numpy array of any shape
    :return A: post-activation parameter A, of the same shape as z
    """
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def linear_backward(dZ: np.array, A_prev: np, W, b):
    """
    Implements the linear portion of backward propagation for a single layer (layer l)
    :param dZ:
    :param A_prev:
    :param W:
    :param b:
    :return:
    """
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def relu_backward(dA: np.array, cache: tuple) -> tuple:
    """
    Implements backward propagation for a single ReLU unit.
    :param dA:
    :param cache:
    :return:
    """
    A_prev, W, b, Z = cache

    dZ = np.array(dA, copy=True)  # convert dz to a numpy array
    dZ[Z <= 0] = 0
    assert dZ.shape == Z.shape

    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)
    return dA_prev, dW, db


def softmax_backward(AL, Y, cache):
    """
    Implements backward propagation for a single Softmax unit.
    :param AL:
    :param Y:
    :param cache:
    :return:
    """
    A_prev, W, b, Z = cache
    dZ = AL - Y

    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)

    return dA_prev, dW, db
