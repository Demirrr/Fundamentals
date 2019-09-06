import numpy as np
from scipy.special import expit
from utilities import softmax, sigmoid_gradient


def neural_network_cost_gradient(parameters, input, output):
    """
    3-layer network cost and gradient function
    :param parameters: pair of (W1, W2)
    :param input: input vector
    :param output: index to correct label
    :return: cross entropy cost and gradient
    """
    W1, W2 = parameters
    input = input.reshape(-1, 1)

    hidden_layer = expit(W1.dot(input))
    inside_softmax = W2.dot(hidden_layer)

    # TODO: allow softmax to normalize column vector
    prediction = softmax(inside_softmax.reshape(-1)).reshape(-1, 1)
    cost = -np.sum(np.log(prediction[output]))

    one_hot = np.zeros_like(prediction)
    one_hot[output] = 1
    delta = prediction - one_hot
    gradient_W2 = delta.dot(hidden_layer.T)
    gradient_W1 = sigmoid_gradient(hidden_layer) * W2.T.dot(delta).dot(input.T)

    gradient = [gradient_W1, gradient_W2]
    return cost, gradient


def flatten_cost_gradient(cost_gradient_hetero, shapes):
    """
    Allow cost function to have heterogeneous parameters (which is not allowed in numpy array)
    :param cost_gradient_hetero: cost function that receives heterogeneous parameters
    :param shapes: list of shapes of parameter
    :return: cost function that receives concatenated parameters and returns concatenated gradients
    """
    def cost_gradient_wrapper(concatenated_parameters, input, output):
        all_parameters = []

        for shape in shapes:
            split_index = np.prod(shape)
            single_parameter, concatenated_parameters = np.split(concatenated_parameters, [split_index])
            single_parameter = single_parameter.reshape(shape)
            all_parameters.append(single_parameter)

        cost, gradients = cost_gradient_hetero(all_parameters, input, output)
        flatten_gradients = [gradient.flatten() for gradient in gradients]
        concatenated_gradients = np.concatenate(flatten_gradients)
        return cost, concatenated_gradients

    return cost_gradient_wrapper
