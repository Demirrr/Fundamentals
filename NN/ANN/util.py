import numpy as np
from enum import Enum

np.seterr(divide="raise")
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def grad_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def grad_relu(z):
    return (z >= 0) * 1


def leakyrelu(z):
    return np.maximum(0.01 * z, z)


def grad_leakyrelu(z):
    return 0.01 * (z < 0) + 1 * (z >= 0)


def grad_tanh(z):
    return 1 - np.tanh(z) ** 2


class ActivationType(Enum):
    sigmoid = "Sigmoid"
    relu = """ReLU"""
    leakyrelu = """LeakyReLU"""
    tanh = """Tanh"""


class ActivationFunction(object):
    def __init__(self, activation_type: ActivationType = ActivationType.relu):
        """
        Class containing the possible activation functions for the nodes of the neural network.
        Select the desired activation function amongst the one provided by the Enum ActivationType

        :param activation_type:
        :type activation_type: ActivationType
        """
        self.__f = None
        self.__grad_f = None

        if type(activation_type) is ActivationType:
            self.__activation_type = activation_type
        else:
            raise ValueError("The activation function must be of the type specified in the Activations Enum")

        if self.__activation_type is ActivationType.sigmoid:
            self.__f = sigmoid
            self.__grad_f = grad_sigmoid
        elif self.__activation_type is ActivationType.relu:
            self.__f = relu
            self.__grad_f = grad_relu
        elif self.__activation_type is ActivationType.leakyrelu:
            self.__f = leakyrelu
            self.__grad_f = grad_leakyrelu
        elif self.__activation_type is ActivationType.tanh:
            self.__f = lambda x: np.tanh(x)
            self.__grad_f = grad_tanh
        else:
            raise ValueError("An Enum is missing")

    def __repr__(self):
        return "{0} activation function".format(self.__activation_type.value)

    @property
    def f(self):
        """
        This function returns the activation function.

        :return:
        """
        return self.__f

    @property
    def grad_f(self):
        """
        This function returns the derivative of the activation function.

        :return:
        """
        return self.__grad_f

    @property
    def activation_type(self):
        """
        Type of activation function

        :return:
        """
        return self.__activation_type


class OptimizationScheme(Enum):
    gradientdescent = "Gradient Descent"
    momentum = "Gradient Descent with momentum"
    adam = "Gradient Descent with adam"


class Regularization(Enum):
    ridge = "L2 regularization"
    lasso = "Lasso"
    dropout = "Dropout"