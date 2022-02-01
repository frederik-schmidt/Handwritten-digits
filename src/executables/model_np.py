""" This file trains a neural network fully implemented in numpy, based on the model
built in the Deep Learning Specialization on Coursera. """
from handwritten_digits.data import *
from handwritten_digits.utils_np import *


def deep_neural_network_np(
    X: np.array,
    Y: np.array,
    layers_dims: list,
    optimizer: str = "MiniBatchGradientDescent",
    alpha: float = 0.1,
    epochs: int = 10,
):
    """
    Trains the neural network.
    :param X: input data, of size (784, m)
    :param Y: true labels, of size (10, m)
    :param layers_dims: network architecture of the neural network
    :param optimizer: optimizer to use for learning
    :param alpha: learning rate
    :param epochs: number of iterations of the optimization loop
    :return: dict containing the learned weights W and biases b
    """
    np.random.seed(1)
    params = initialize_params(layers_dims)

    if optimizer == "GradientDescent":
        for epoch in range(epochs):
            AL, caches = forward_propagation(X, params)
            grads = backward_propagation(AL, Y, caches)
            params = update_params(params, grads, alpha)

            if epoch % 100 == 0 or epoch == epochs - 1:
                cost = compute_cross_entropy_cost(AL, Y)
                preds, _ = predict(X, params)
                accuracy = compute_accuracy(preds, Y)
                print(f"--- Epoch: {epoch} ---")
                print("Cost:", cost)
                print("Accuracy:", accuracy)

    elif optimizer == "MiniBatchGradientDescent":
        mini_batches = random_mini_batches(X, Y, mini_batch_size=64)
        for epoch in range(1, epochs):
            for mini_batch in mini_batches:
                X_mini, Y_mini = mini_batch
                AL, caches = forward_propagation(X_mini, params)
                grads = backward_propagation(AL, Y_mini, caches)
                params = update_params(params, grads, alpha)

            AL, _ = forward_propagation(X, params)
            cost = compute_cross_entropy_cost(AL, Y)
            preds, _ = predict(X, params)
            accuracy = compute_accuracy(preds, Y)
            print(f"--- Epoch: {epoch} ---")
            print("Cost:", cost)
            print("Accuracy:", accuracy)

    return params


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_prepare_mnist_data()
    X_train, y_train = X_train[:, 0:5000], y_train[:, 0:5000]
    layers_dims = [784, 10, 10]
    params = deep_neural_network_np(
        X=X_train,
        Y=y_train,
        layers_dims=layers_dims,
        optimizer="MiniBatchGradientDescent",
        alpha=0.1,
        epochs=10,
    )
