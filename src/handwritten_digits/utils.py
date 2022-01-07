import numpy as np
from handwritten_digits.activation_functions import softmax, relu, relu_backward
from handwritten_digits.data import one_hot


def initialize_params(layer_dims: list) -> dict:
    """
    Initializes weights and biases according to the dimensions of each layer.
    :param layer_dims: dimensions of each layer of the neural network
    :return: parameters W and b
    """
    np.random.seed(1)
    L = len(layer_dims)  # number of layers
    params = {}

    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return params


def forward_propagation(X: np.array, params: dict):
    """
    Implements forward propagation.
    :param X: activations of previous layer (or input data)
    :param params: parameters W and b
    :return: output of forward propagation Z and A
    """
    L = len(params) // 2  # number of layers
    activations = {}

    for l in range(1, L):
        A_prev = X if l == 1 else activations["A" + str(l - 1)]
        activations["Z" + str(l)] = (np.dot(params["W" + str(l)], A_prev) + params["b" + str(l)])
        activations["A" + str(l)] = relu(activations["Z" + str(l)])

    activations["Z" + str(L)] = (np.dot(params["W" + str(L)], activations["A" + str(L - 1)]) + params["b" + str(L)])
    activations["A" + str(L)] = softmax(activations["Z" + str(L)])

    return activations


def backward_propagation(X: np.array, Y: np.array, params: dict, activations: dict) -> dict:
    """
    Implements backward propagation.
    :param X: activations of forward propagation
    :param Y: true label
    :param params: parameters W and b
    :param activations: output of forward propagation Z and A
    :return: gradients
    """
    L = len(params) // 2  # number of layers
    grads = {}
    m = Y.size
    one_hot_Y = one_hot(Y)
    assert activations["A" + str(L)].shape == one_hot_Y.shape

    grads["dZ" + str(L)] = activations["A" + str(L)] - one_hot_Y
    grads["dW" + str(L)] = (1 / m * np.dot(grads["dZ" + str(L)], activations["A" + str(L - 1)].T))
    grads["db" + str(L)] = 1 / m * np.sum(grads["dZ" + str(L)])

    for l in reversed(range(1, L)):
        grads["dZ" + str(l)] = np.dot(params["W" + str(l + 1)].T, grads["dZ" + str(l + 1)]) * relu_backward(activations["Z" + str(l)])
        grads["dW" + str(l)] = 1 / m * np.dot(grads["dZ" + str(l)], X.T)  # TODO: X.T
        grads["db" + str(l)] = 1 / m * np.sum(grads["dZ" + str(l)])

    return grads


def update_params(params: dict, grads: dict, alpha: float) -> dict:
    """
    Updates the parameters.
    :param params: initial parameters W and b
    :param grads: gradients
    :param alpha: learning rate
    :return: updated parameters W and b
    """
    L = len(params) // 2  # number of layers

    for l in range(L):
        params["W" + str(l + 1)] -= alpha * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] -= alpha * grads["db" + str(l + 1)]

    return params


def get_predictions(AL: np.array) -> np.array:
    """
    Gets the predicted class.
    :param AL: label predictions
    :return: index of the class with the highest predictions
    """
    pred = np.argmax(AL, axis=0)
    return pred


def compute_accuracy(predictions: np.array, Y: np.array) -> float:
    """
    Computes the accuracy of the predictions with respect to the true labels.
    :param predictions: label predictions
    :param Y: true labels
    :return: accuracy between 0 and 1
    """
    print(f"First 10 true labels: {Y[:10]}")
    print(f"First 10 predictions: {predictions[:10]}")
    accuracy = np.sum(predictions == Y) / Y.size
    return accuracy
