from scipy.ndimage.interpolation import shift
import numpy as np


def augment_data(X_train: np.array, y_train: np.array, shifts: tuple) -> tuple:
    """
    Performs data augmentation by shifting the input images.
    :param X_train:
    :param y_train:
    :param shifts:
    :return X_train_aug:
    :return y_train_aug:
    """
    X_train_aug = X_train.copy()
    y_train_aug = y_train.copy()
    for shift_x, shift_y in shifts:
        for image, label in zip(X_train, y_train):
            shifted_image = shift(image, shift=[shift_y, shift_x]).reshape((1, 28, 28))
            X_train_aug = np.vstack([X_train_aug, shifted_image])
            y_train_aug = np.append(y_train_aug, label)
    return X_train_aug, y_train_aug
