""" This file trains a neural network fully implemented in numpy, based on the model
built in the Deep Learning Specialization on Coursera. """
from handwritten_digits.data import *
from handwritten_digits.utils_np import *


def deep_neural_network_np(
    X: np.array,
    Y: np.array,
    layers_dims: list,
    alpha: float = 0.0075,
    num_iterations: int = 3000,
    print_cost: bool = False,
):
    """
    Trains the neural network using gradient descent.
    :param X: input data
    :param Y: true labels
    :param layers_dims: network architecture of the neural network
    :param alpha: learning rate
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: if set to True, cost will be printed every 100 iterations
    :return: dict containing the learned weights W and biases b
    """
    np.random.seed(1)
    costs = []

    params = initialize_params(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = forward_propagation(X, params)
        cost = compute_cross_entropy_cost(AL, Y)
        grads = backward_propagation(AL, Y, caches)
        params = update_params(params, grads, alpha)

        if print_cost and i % 100 == 0:
            print(f"--- Iteration: {i} ---")
            print("Cost:", cost)
        if print_cost and i % 10 == 0:
            costs.append(cost)

    return params


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_prepare_mnist_data()
    X_train, y_train = X_train[:, 0:5000], y_train[:, 0:5000]
    layers_dims = [784, 10, 10]
    params = deep_neural_network_np(
        X=X_train,
        Y=y_train,
        layers_dims=layers_dims,
        alpha=0.1,
        num_iterations=500,
        print_cost=True,
    )
