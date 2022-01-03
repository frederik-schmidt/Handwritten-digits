import numpy as np


def sigmoid(z: np.array) -> tuple:
    """
    Implements the sigmoid activation function in numpy.
    :param z: numpy array of any shape
    :return a: post-activation parameter, of the same shape as z
    :return cache: dict containing a, stored for computing the backward pass efficiently
    """
    a = 1 / (1 + np.exp(-z))
    cache = z
    return a, cache


def relu(z: np.array) -> tuple:
    """
    Implements the RELU activation function in numpy.
    :param z: numpy array of any shape
    :return a: post-activation parameter, of the same shape as z
    :return cache: dict containing a, stored for computing the backward pass efficiently
    """
    a = np.maximum(0, z)
    cache = z
    return a, cache


def softmax(z: np.array) -> tuple:
    """
    Implements the softmax activation function in numpy.
    :param z: numpy array of any shape
    :return a: post-activation parameter, of the same shape as z
    :return cache: dict containing a, stored for computing the backward pass efficiently
    """
    exp_z = np.exp(z - np.max(z))
    a = exp_z / exp_z.sum(axis=0)
    cache = z
    return a, cache
