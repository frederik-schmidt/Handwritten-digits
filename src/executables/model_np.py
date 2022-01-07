from handwritten_digits.data import load_and_prepare_mnist_data
from handwritten_digits.utils import *


def nn_model_np(
        X: np.array,
        Y: np.array,
        layer_dims: list,
        num_iterations: int,
        alpha: float,
        print_cost: bool = True
) -> dict:
    """
    Trains the model using gradient descent optimizer.
    :param X: input images
    :param Y: true labels
    :param layer_dims: dimensions of each layer of the neural network
    :param num_iterations: number of iterations to train
    :param alpha: learning rate of the model
    :param print_cost: flag indicating whether or not to print the costs
    :return: parameters W and b
    """
    L = len(layer_dims)  # number of layers
    params = initialize_params(layer_dims=layer_dims)
    for i in range(num_iterations):
        activations = forward_propagation(X, params)
        grads = backward_propagation(X, Y, params, activations)
        params = update_params(params, grads, alpha)
        if print_cost and i % 50 == 0 or i == num_iterations - 1:
            print(f"--- Iteration: {i} ---")
            preds = get_predictions(activations["A" + str(L - 1)])
            print("Accuracy:", compute_accuracy(preds, Y))
    return params


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_prepare_mnist_data()
    params = nn_model_np(
        X=X_train,
        Y=y_train,
        layer_dims=[784, 20, 10],
        num_iterations=300,
        alpha=0.1,
    )
