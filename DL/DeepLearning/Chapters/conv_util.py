import numpy as np
import matplotlib.pyplot as plt

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    return X_pad


def select(X, ith_slice, rec_shape=None):
    """
    Select rectange
    Argument:
    X --
    ith_slice --
    rec_shape --
    Returns:
    X_select --
    """
    # rec_shape denote a rectangle shape.
    # a--c
    # b--d
    if rec_shape:
        a, b, c, d = rec_shape
        return X[ith_slice, a:b, c:d, :]
    else:
        _, b, d, _ = X.shape
        return X[ith_slice, 0:b, 0:d, :]


def show_slice(x):
    plt.imshow(x, interpolation='none')
    plt.show()


#X_padded = zero_pad(X, 1)
#show_slice(select(X_padded, 0))
#show_slice(select(X_padded, 0, rec_shape=(0, 4, 0, 4)))
#show_slice(select(X_padded, 0, rec_shape=(0, 4, 6, 10)))