import numpy as np


def sigmoid(Z: np.array) -> tuple:
    """
    Implements the forward propagation for A single sigmoid unit.
    :param Z: numpy array of any shape
    :return A: numpy array containing post-activation parameter A, of the same shape as z
    :return cache: numpy array containing z, stored for computing the backward pass efficiently
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z: np.array) -> tuple:
    """
    Implements the forward propagation for A single RELU unit.
    :param Z: numpy array of any shape
    :return A: numpy array containing post-activation parameter A, of the same shape as z
    :return cache: numpy array containing z, stored for computing the backward pass efficiently
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def softmax(Z: np.array) -> tuple:
    """
    Implements the forward propagation for A single softmax unit.
    :param Z: numpy array of any shape
    :return A: post-activation parameter A, of the same shape as z
    :return cache: numpy array containing z, stored for computing the backward pass efficiently
    """
    exp_Z = np.exp(Z - np.max(Z))
    A = exp_Z / exp_Z.sum(axis=0)
    cache = Z
    return A, cache


def sigmoid_backward(dA: np.array, cache: np.array) -> np.array:
    """
    Implements the backward propagation for a single sigmoid unit in numpy.
    :param dA: post-activation gradient da, of any shape
    :param cache: numpy array containing Z, stored for computing the backward pass efficiently
    :return dZ: gradient of the cost with respect to Z
    """
    Z = cache
    s, _ = sigmoid(Z)
    dZ = dA * s * (1 - s)
    assert dZ.shape == Z.shape
    return dZ


def relu_backward(dA: np.array, cache: np.array) -> np.array:
    """
    Implements the backward propagation for a single RELU unit in numpy.
    :param dA: post-activation gradient da, of any shape
    :param cache: numpy array containing Z, stored for computing the backward pass efficiently
    :return dZ: gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert dZ.shape == Z.shape
    return dZ


def softmax_backward(dA: np.array, cache: np.array) -> np.array:
    """
    Implements the backward propagation for a single softmax unit in numpy.
    :param dA: post-activation gradient da, of any shape
    :param cache: numpy array containing Z, stored for computing the backward pass efficiently
    :return dZ: gradient of the cost with respect to Z
    """
    Z = cache
    s, _ = softmax(dA)
    dZ = s / np.sum(s, axis=0) * (1 - s / np.sum(s, axis=0))
    assert dZ.shape == Z.shape
    return dZ
