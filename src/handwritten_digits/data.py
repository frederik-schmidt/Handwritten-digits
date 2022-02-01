import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
from tensorflow.keras.datasets import mnist


def one_hot(Y: np.array) -> np.array:
    """
    Performs one hot encoding for a given array.
    :param Y: true labels, of size (m)
    :return: one hot true encoded labels, of size (10, m)
    >>> Y_t = np.array([1,9])
    >>> one_hot(Y_t)
    array([[0., 0.],
           [1., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.]])
    """
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T


def load_and_prepare_mnist_data():
    """
    Loads the mnist dataset from Tensorflow, reshapes and standardizes the arrays
    X_train and X_test which contain the images with handwritten digits, one
    hot encodes the labels y_train and y_test.
    :return: tuple containing prepared input images and labels for training and testing
    """
    (X_train_orig, y_train_orig), (X_test_orig, y_test_orig) = mnist.load_data()
    X_train_flat = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_train = X_train_flat / 255.0
    X_test_flat = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    X_test = X_test_flat / 255.0
    y_train, y_test = one_hot(y_train_orig), one_hot(y_test_orig)
    return X_train, y_train, X_test, y_test


def random_mini_batches(X: np.array, Y: np.array, mini_batch_size: int = 64) -> list:
    """
    Creates random mini-batches from input data and true labels.
    :param X: input data, of size (784, m)
    :param Y: true labels, of size (m)
    :param mini_batch_size: number of training examples per mini-batch
    :return: list containing a tuple with X and Y for every mini-batch
    """
    m = X.shape[1]  # number of training examples
    mini_batches = []
    # Shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    # Partition
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handle end case
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def augment_data(X_train: np.array, y_train: np.array, shifts: tuple) -> tuple:
    """
    Performs data augmentation by shifting the input images.
    :param X_train: initial input images
    :param y_train: initial true labels, of size (m)
    :param shifts: tuple of tuples indicating the shifts on x-axis and y-axis, e.g. shifts=((1,0),(-1,0))
    :return X_train_aug: augmented input images, of size (784, m)
    :return y_train_aug: true labels, of size (10, m)
    """
    X_train_aug = X_train.copy()
    y_train_aug = y_train.copy()
    for shift_x, shift_y in shifts:
        for image, label in zip(X_train, y_train):
            shifted_image = shift(image, shift=[shift_y, shift_x]).reshape((1, 28, 28))
            X_train_aug = np.vstack([X_train_aug, shifted_image])
            y_train_aug = np.append(y_train_aug, label)
    return X_train_aug, y_train_aug


def plot_digit(X: np.array, Y: np.array, index: int) -> plt.imshow:
    """
    Plots a digit.
    :param X: input images, of size (784, m)
    :param Y: true labels, of size (10, m)
    :param index: index of the input image to be plotted
    :return: plotted input image
    """
    image = X[:, index].reshape(28, 28)
    label = np.argmax(Y[:, index])
    plt.imshow(image, cmap='Greys')
    plt.title("Y = " + str(label))
