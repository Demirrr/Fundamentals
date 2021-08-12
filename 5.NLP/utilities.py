import numpy as np


def softmax(x):
    """
    Normalize any vector to probabilistic distribution.
    :param x: numpy array or matrix
    :return: numpy array or matrix of the same shape to x
    """
    xmax = np.expand_dims(np.max(x, -1), -1)
    e_x = np.exp(x - xmax)
    x = e_x / np.expand_dims(np.sum(e_x, -1), -1)
    return x


def sigmoid_gradient(f):
    """
    Sigmoid gradient function
    :param f: function value of sigmoid function
    :return: gradient value of sigmoid function
    """
    gradient = f * (1.0 - f)
    return gradient


def tanh_gradient(f):
    """
    Tanh gradient function
    :param f: function value of tanh
    :return: gradient value of tanh
    """
    gradient = 1 - f ** 2
    return gradient
