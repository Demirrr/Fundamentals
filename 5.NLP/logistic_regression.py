import numpy as np
from scipy.special import expit
from utilities import softmax



def logistic_regression_cost_gradient(parameters, input, output):
    """
    Cost and gradient for logistic regression
    :param parameters: weight vector
    :param input: feature vector
    :param output: binary label (0 or 1)
    :return: cost and gradient for the input and output
    """
    prediction = expit(np.dot(input, parameters))
    if output:
        inside_log = prediction
    else:
        inside_log = 1.0 - prediction

    if inside_log != 0.0:
        cost = -np.log(inside_log)
    else:
        cost = np.finfo(float).min

    gradient = (prediction - output) * input
    return cost, gradient


def multinomial_logistic_regression_cost_gradient(parameters, input, output):
    """
    Cost and gradient for multinomial logistic regression
    :param parameters: weight vector
    :param input: feature vector
    :param output: integer label
    :return: cost and gradient for the input and output
    """
    prediction = softmax(np.dot(parameters.T, input))
    cost = -np.log(prediction[output])
    # Create one-hot vector
    one_hot = np.zeros_like(prediction)
    one_hot[output] = 1
    gradient = np.dot(input.reshape(-1, 1), (prediction - one_hot).reshape(-1, 1).T)
    return cost, gradient
