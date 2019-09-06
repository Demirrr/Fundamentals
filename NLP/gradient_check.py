import numpy as np


def get_numeric_gradient(function_gradient, parameters, index, epsilon):
    """
    Get numeric gradient of function at specific index of parameters
    :param function_gradient: any function that takes parameters and outputs function value and its gradients
    :param parameters: the point of numpy array to check the gradient at
    :param index: the dimension of parameters to compute numeric gradient
    :param epsilon: small number to compute numeric gradient
    :return: numeric gradient of function at the index of parameters
    """
    parameters_step = parameters.copy()
    parameters_step[index] += epsilon / 2
    np.random.seed(0)
    function_value_positive, _ = function_gradient(parameters_step)

    parameters_step = parameters.copy()
    parameters_step[index] -= epsilon / 2
    np.random.seed(0)
    function_value_negative, _ = function_gradient(parameters_step)
    numeric_gradient = (function_value_positive - function_value_negative) / epsilon
    return numeric_gradient


def gradient_check(function_gradient, parameters, epsilon=1e-4, threshold=1e-5):
    """
    Check gradient of any function by comparing numeric gradient
    :param function_gradient: function that takes single argument and outputs function value and its gradients
    :param parameters: the point (numpy array) to check the gradient at
    :param epsilon: small number to compute numerical gradient (optional)
    :param threshold: threshold to fail gradient check (optional)
    :return: list of tuple about information at failed point (empty list if passed)
    """
    np.random.seed(0)
    function_value, gradient = function_gradient(parameters)     # Evaluate function value at original point

    # Iterate over all indices in parameters of multi-dimensional array
    iterator = np.nditer(parameters, flags=['multi_index'], op_flags=['readwrite'])

    result = []
    while not iterator.finished:
        index = iterator.multi_index

        numeric_gradient = get_numeric_gradient(function_gradient, parameters, index, epsilon)
        difference = abs(numeric_gradient - gradient[index]) / max(1, abs(numeric_gradient), abs(gradient[index]))

        if difference > threshold:
            # Gradient check failed at index!
            result.append((index, gradient[index], numeric_gradient))

        iterator.iternext()

    return result
