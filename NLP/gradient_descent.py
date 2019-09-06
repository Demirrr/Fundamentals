import numpy as np


def get_constant(learning_rate=0.5):
    """
    Constant learning rate for gradient descent
    """
    def constant(gradient):
        return learning_rate
    return constant


def get_adagrad(learning_rate=0.5):
    """
    Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
    John Duchi, Elad Hazan and Yoram Singer, Journal of Machine Learning Research 12 (2011) 2121-2159
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """
    sum_square_gradient = None

    def adagrad(gradient):
        nonlocal sum_square_gradient

        if sum_square_gradient is None:
            sum_square_gradient = np.ones_like(gradient)
        sum_square_gradient += gradient ** 2
        return learning_rate / np.sqrt(sum_square_gradient)

    return adagrad


def gradient_descent(cost_gradient, initial_parameters, iterations=1000, learning_rate=get_constant()):
    """
    Gradient Descent finds parameters that minimizes cost function
    :param cost_gradient: function to get cost and gradient given parameters
    :param initial_parameters: the initial point to start gradient descent
    :param iterations: total iterations to run gradient descent
    :param learning_rate: algorithm to get and update learning rate
    :return: final parameters and history of cost
    """
    parameters = initial_parameters
    cost_history = []

    for i in range(iterations):
        cost, gradient = cost_gradient(parameters)

        # Stop update if cost is not improved anymore
        if len(cost_history) > 0 and cost_history[-1] == cost:
            continue

        step = learning_rate(gradient)
        parameters -= step * gradient
        cost_history.append(cost)

    return parameters, cost_history
