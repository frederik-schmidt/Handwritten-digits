import numpy as np


def initialize_params(layer_dims: list) -> dict:
    """
    Initializes weights and biases according to the dimensions of each layer.
    :param layer_dims: dimensions of each layer of the neural network
    :return: dict containing the weight matrix W and bias vector b
    >>> initialize_params(layer_dims=[2,1])
    {'W1': array([[ 0.01624345, -0.00611756]]), 'b1': array([[0.]])}
    """
    np.random.seed(1)
    L = len(layer_dims)  # number of layers
    params = {}

    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return params


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


def linear_activation_forward(
    A_prev: np.array, W: np.array, b: np.array, activation: str
) -> tuple:
    """
    Implements forward propagation for a single layer.
    :param A_prev: gradient of cost with respect to A (of previous layer l-1), same shape as A_prev
    :param W: gradient of cost with respect to W (current layer l), same shape as W
    :param b: gradient of cost with respect to b (current layer l), same shape as b
    :param activation: activation function to be used
    :return A: post-activation parameter A
    :return cache: cached values for computing backward pass efficiently
    """
    Z = np.dot(W, A_prev) + b

    if activation == "relu":
        A = relu(Z)

    elif activation == "softmax":
        A = softmax(Z)

    else:
        raise ValueError(f"activation '{activation}' not supported.")

    cache = (A_prev, W, b, Z)
    return A, cache


def forward_propagation(X: np.array, params: np.array) -> tuple:
    """
    Implements forward propagation over all layers.
    :param X: input data or activations of previous layer
    :param params: dict containing the weight matrix W and bias vector b
    :return AL: output of forward pass AL
    :return caches: cached values for computing backward pass efficiently
    """
    caches = []
    A = X
    L = len(params) // 2  # number of hidden layers

    for l in range(1, L):
        A_prev, W, b = A, params["W" + str(l)], params["b" + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, activation="relu")
        caches.append(cache)

    W, b = params["W" + str(L)], params["b" + str(L)]
    AL, cache = linear_activation_forward(A, W, b, activation="softmax")

    caches.append(cache)

    assert AL.shape == (10, X.shape[1])
    return AL, caches


def compute_cross_entropy_cost(AL: np.array, Y: np.array) -> float:
    """
    Implements the cross-entropy cost function.
    :param AL: label predictions
    :param Y: true labels
    :return: cross-entropy cost
    """
    m = Y.size
    cost = -1 / m * np.sum(np.multiply(Y, np.log(AL)))
    cost = np.squeeze(cost)  # this turns e.g. [[17]] into 17
    return cost


def linear_backward(dZ: np.array, A_prev: np, W, b):
    """
    Implements the linear portion of backward propagation for a single layer (layer l)
    :param dZ: gradient of the cost with respect to the linear output (of current layer l)
    :param A_prev: activations from previous layer (or input data)
    :param W: weight matrix W
    :param b: bias vector b
    :return dA_prev: gradient of cost with respect to the activation (previous layer l-1)
    :return dW: gradient of cost with respect to W (current layer l)
    :return db: gradient of the cost with respect to b (current layer l)
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
    :param dA: post-activation gradient for current layer l
    :param cache: cached values for computing backward pass efficiently
    :return dA_prev: gradient of cost with respect to the activation (previous layer l-1)
    :return dW: gradient of cost with respect to W (current layer l)
    :return db: gradient of the cost with respect to b (current layer l)
    """
    A_prev, W, b, Z = cache

    dZ = np.array(dA, copy=True)  # convert dz to a numpy array
    dZ[Z <= 0] = 0
    assert dZ.shape == Z.shape

    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)
    return dA_prev, dW, db


def softmax_backward(AL: np.array, Y: np.array, cache: tuple):
    """
    Implements backward propagation for a single Softmax unit.
    :param AL: post-activation value of last layer L
    :param Y: true labels
    :param cache: cached values for computing backward pass efficiently
    :return dA_prev: gradient of cost with respect to the activation (previous layer l-1)
    :return dW: gradient of cost with respect to W (current layer l)
    :return db: gradient of the cost with respect to b (current layer l)
    """
    A_prev, W, b, _ = cache
    dZ = AL - Y

    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)

    return dA_prev, dW, db


def backward_propagation(AL: np.array, Y: np.array, caches: list) -> dict:
    """
    Implements backward propagation over all layers.
    :param AL: post-activation value of last layer L
    :param Y: true labels
    :param caches: cached values from forward pass
    :return: gradients
    """
    grads = {}
    L = len(caches)  # number of layers
    Y = Y.reshape(AL.shape)

    cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = softmax_backward(AL, Y, cache)

    for l in reversed(range(L - 1)):
        cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = relu_backward(grads["dA" + str(l + 1)], cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_params(params: dict, grads: dict, alpha: float) -> dict:
    """
    Updates the parameters using gradient descent.
    :param params: dict containing the initial weight matrix W and bias vector b
    :param grads: gradients
    :param alpha: learning rate
    :return: dict containing the updated weight matrix W and bias vector b
    """
    L = len(params) // 2  # number of layers

    for l in range(L):
        params["W" + str(l + 1)] -= alpha * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] -= alpha * grads["db" + str(l + 1)]

    return params


def predict(X: np.array, params: dict) -> tuple:
    """
    Predicts the results of a L-layer neural network.
    :param X: input data
    :param params: dict containing the learned weight matrix W and bias vector b
    :return preds: label predictions
    :return probs:
    """
    probs, caches = forward_propagation(X, params)

    # highest probability for a given example is coded as 1, otherwise 0
    preds = probs == np.amax(probs, axis=0, keepdims=True)
    preds = preds.astype(float)

    return preds, probs
