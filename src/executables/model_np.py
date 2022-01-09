from handwritten_digits.data import load_and_prepare_mnist_data, one_hot
from handwritten_digits.utils_np import *


def deep_neural_network_np(
    X: np.array,
    Y: np.array,
    layers_dims: list,
    learning_rate: float = 0.0075,
    num_iterations: int = 3000,
    print_cost: bool = False,
):
    """
    Trains the neural network using gradient descent.
    :param X:
    :param Y:
    :param layers_dims:
    :param learning_rate:
    :param num_iterations:
    :param print_cost:
    :return:
    """
    np.random.seed(1)
    costs = []

    params = initialize_params(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = forward_propagation(X, params)
        cost = compute_cross_entropy_cost(AL, Y)
        grads = backward_propagation(AL, Y, caches)
        params = update_params(params, grads, learning_rate)

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
        learning_rate=0.001,
        num_iterations=500,
        print_cost=True,
    )
