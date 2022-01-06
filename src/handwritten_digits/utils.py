import numpy as np
import activations


def initialize_params(layer_dims: list) -> dict:
    """
    Initializes weights and biases according to dimensions of each layer.
    :param layer_dims: list containing the dimensions of each layer
    :return: dict containing the params W and b
    """
    params = {}
    L = len(layer_dims)
    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return params


def linear_forward(A: np.array, W: np.array, b: np.array) -> tuple:
    """
    Implements the linear part of a layer's forward propagation.
    :param A: activations from previous layer (or input data)
    :param W: weights matrix of shape (size of current layer, size of previous layer)
    :param b: bias vector of shape (size of the current layer, 1)
    :return Z: output of linear forward pass
    :return cache: tuple containing a, w and b, stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(
    A_prev: np.array, W: np.array, b: np.array, activation: str
) -> tuple:
    """
    Implements the forward propagation for the LINEAR->ACTIVATION layer.
    :param A_prev: activations from previous layer (or input data)
    :param W: weights matrix of shape (size of current layer, size of previous layer)
    :param b: bias vector of shape (size of the current layer, 1)
    :param activation: name of the activation function, 'sigmoid', 'relu' or 'softmax'
    :return A: output of the activation function Z
    :return cache: cached 'A', 'w' and 'b', stored for computing the backward pass efficiently
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = activations.sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = activations.relu(Z)

    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = activations.softmax(Z)

    else:
        raise ValueError(f"activation '{activation}' is not supported.")

    cache = (linear_cache, activation_cache)
    return A, cache


def linear_backward(dZ: np.array, cache: tuple) -> tuple:
    """
    Implements the linear portion of backward propagation for a single layer (layer l).
    :param dZ: gradient of the cost with respect to the linear output (of current layer l)
    :param cache: tuple of values (a_prev, W, b) coming from the forward propagation
    :return dA_prev: gradient of the cost with respect to the activation (of previous layer l-1), same shape as a_prev
    :return dW: gradient of the cost with respect to W (of current layer l), same shape as W
    :return db: gradient of the cost with respect to b (of current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dw = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dw, db


def linear_activation_backward(dA, cache, activation) -> tuple:
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer.

    :param dZ: gradient of the cost with respect to the linear output (of current layer l)
    :param cache: tuple of values (A_prev, W, b) coming from the forward propagation
    :return dA_prev: gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    :return dW: gradient of the cost with respect to W (current layer l), same shape as W
    :return db: gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = activations.sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "relu":
        dZ = activations.relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "softmax":
        dZ = activations.softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    else:
        raise ValueError(f"activation '{activation}' is not supported.")

    return dA_prev, dW, db


def L_model_forward(X: np.array, parameters: np.array) -> tuple:
    """
    Implements forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    :param X: data of shape (input size, number of examples)
    :param parameters: output of initialize_parameters_deep()
    :return: activation value from the output (last) layer and list of L caches from linear_activation_forward()
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev=A_prev,
            W=parameters["W" + str(l)],
            b=parameters["b" + str(l)],
            activation="relu",
        )
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A_prev=A,
        W=parameters["W" + str(L)],
        b=parameters["b" + str(L)],
        activation="softmax",
    )
    caches.append(cache)

    return AL, caches


def L_model_backward(AL: np.array, Y: np.array, caches: np.array) -> dict:
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    :param AL: probability vector with label predictions
    :param Y: true label vector
    :param caches: list containing cache for each layer
    :return grads: dict containing gradients
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initialize the backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL

    # Lth layer (SOFTMAX -> LINEAR) gradients
    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
        dA=dAL,
        cache=current_cache,
        activation="softnax",
    )
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            dA=dA_prev_temp,
            cache=current_cache,
            activation="relu",
        )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(params: dict, grads: dict, learning_rate: float) -> dict:
    """
    Updates parameters using gradient descent.
    :param params: parameters
    :param grads: gradients, output of L_model_backward
    :param learning_rate: learning rate alpha
    :return params: updated parameters
    """
    parameters = params.copy()
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = (
            parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        )
        parameters["b" + str(l + 1)] = (
            parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        )
    return parameters
