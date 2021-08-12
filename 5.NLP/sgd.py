import numpy as np


def batch_sampler(size):
    """
    Helper function for batch gradient descent
    :param size: size of data set
    :return: all indices in data set
    """
    return range(size)


def get_stochastic_sampler(batch_size):
    """
    Helper function for stochastic gradient descent
    :param batch_size: size of mini batch
    :return: function that return indices of batch size sampled from data set
    """
    def stochastic_sampler(size):
        indices = np.random.randint(0, size, batch_size)
        return indices
    return stochastic_sampler


def get_shufle_sampler(size):
    """
    Shuffle original data and iterate over it repeatedly
    :param size: size of data
    :return: function that return index
    """
    indices = np.arange(size)
    np.random.shuffle(indices)
    i = 0

    def shuffle_sampler(size):
        nonlocal i
        i += 1
        i %= size
        return [indices[i]]
    return shuffle_sampler


def bind_cost_gradient(cost_gradient_sample, inputs, outputs=None, sampler=batch_sampler):
    """
    Bind per-sample cost and gradient function to data set for gradient descent
    :param cost_gradient_sample: function to get cost and gradient for pair of input and output
    :param inputs: feature vectors of data set
    :param outputs: labels of data set for supervised learning
    :param sampler: either batch or stochastic sampler to get sample from data set
    :return: function that receives parameters and return cost and gradient
    """
    def cost_gradient_wrapper(parameters):
        indices = sampler(len(inputs))

        total_cost = 0.0
        total_gradient = None

        for index in indices:
            input = inputs[index]

            if outputs is not None:
                output = outputs[index]
            else:
                output = None

            cost, gradient = cost_gradient_sample(parameters, input, output)

            if total_gradient is None:
                total_gradient = np.zeros_like(gradient)

            total_cost += cost
            total_gradient += gradient

        total_cost /= len(indices)
        total_gradient /= len(indices)

        return total_cost, total_gradient

    return cost_gradient_wrapper

