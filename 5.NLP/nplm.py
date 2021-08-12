import numpy as np
from utilities import softmax, tanh_gradient
from preprocessing import build_dictionary, to_indices
from gradient_check import gradient_check
from gradient_descent import gradient_descent
from sgd import bind_cost_gradient, get_stochastic_sampler
from neural_network import flatten_cost_gradient


class NPLM:
    """
    Neural Probabilistic Language Model (Bengio 2003)
    """
    def __init__(self, vocabulary_size, feature_size, context_size, hidden_size):
        self.vocabulary_size = vocabulary_size
        self.feature_size = feature_size
        self.context_size = context_size
        self.hidden_size = hidden_size

        self.W_shape = (vocabulary_size + 1, feature_size * context_size + 1)
        self.U_shape = (vocabulary_size + 1, hidden_size)
        self.H_shape = (hidden_size, feature_size * context_size + 1)
        self.C_shape = (vocabulary_size + 1, feature_size)

        self.dictionary = None
        self.reverse_dictionary = None
        self.parameters = None

    def train(self, sentences, iterations=1000):
        # Preprocess sentences to create indices of context and next words
        self.dictionary = build_dictionary(sentences, self.vocabulary_size)
        indices = to_indices(sentences, self.dictionary)
        self.reverse_dictionary = {index: word for word, index in self.dictionary.items()}
        inputs, outputs = self.create_context(indices)

        # Create cost and gradient function for gradient descent
        shapes = [self.W_shape, self.U_shape, self.H_shape, self.C_shape]
        flatten_nplm_cost_gradient = flatten_cost_gradient(nplm_cost_gradient, shapes)
        cost_gradient = bind_cost_gradient(flatten_nplm_cost_gradient, inputs, outputs,
                                           sampler=get_stochastic_sampler(10))

        # Train neural network
        parameters_size = np.sum(np.product(shape) for shape in shapes)
        initial_parameters = np.random.normal(size=parameters_size)
        self.parameters, cost_history = gradient_descent(cost_gradient, initial_parameters, iterations)
        return cost_history

    def predict(self, context):
        if self.dictionary is None or self.parameters is None:
            print('Train before predict!')
            return
        context = context[-self.context_size:]
        input = []
        for word in context:
            if word in self.dictionary:
                input.append(self.dictionary[word])
            else:
                input.append(0)
        W_size = np.product(self.W_shape)
        U_size = np.product(self.U_shape)
        H_size = np.product(self.H_shape)
        split_indices = [W_size, W_size + U_size, W_size + U_size + H_size]
        W, U, H, C = np.split(self.parameters, split_indices)
        W = W.reshape(self.W_shape)
        U = U.reshape(self.U_shape)
        H = H.reshape(self.H_shape)
        C = C.reshape(self.C_shape)

        x = np.concatenate([C[input[i]] for i in range(self.context_size)])
        x = np.append(x, 1.)    # Append bias term
        x = x.reshape(-1, 1)
        y = W.dot(x) + U.dot(np.tanh(H.dot(x)))

        # You don't want to predict unknown words (index 0)
        prediction = np.argmax(y[1:]) + 1
        return self.reverse_dictionary[prediction]

    def create_context(self, sentences):
        inputs = []
        outputs = []
        for sentence in sentences:
            context = []
            for word in sentence:
                if len(context) >= self.context_size:
                    context = context[-self.context_size:]
                    inputs.append(context)
                    outputs.append(word)
                context = context + [word]
        return inputs, outputs

    def gradient_check(self, inputs, outputs):
        # Create cost and gradient function for gradient check
        shapes = [self.W_shape, self.U_shape, self.H_shape, self.C_shape]
        flatten_nplm_cost_gradient = flatten_cost_gradient(nplm_cost_gradient, shapes)
        cost_gradient = bind_cost_gradient(flatten_nplm_cost_gradient, inputs, outputs)

        # Gradient check!
        parameters_size = np.sum(np.product(shape) for shape in shapes)
        initial_parameters = np.random.normal(size=parameters_size)
        result = gradient_check(cost_gradient, initial_parameters)
        return result


def nplm_cost_gradient(parameters, input, output):
    """
    Cost function for NPLM
    :param parameters: tuple of (W, U, H, C)
    :param input: indices of context word
    :param output: index of current word
    :return: cost and gradient
    """
    W, U, H, C = parameters
    context_size = len(input)
    x = np.concatenate([C[input[i]] for i in range(context_size)])
    x = np.append(x, 1.)    # Append bias term
    x = x.reshape(-1, 1)
    hidden_layer = np.tanh(H.dot(x))
    y = W.dot(x) + U.dot(hidden_layer)
    prediction = softmax(y.reshape(-1)).reshape(-1, 1)
    cost = -np.sum(np.log(prediction[output]))

    one_hot = np.zeros_like(prediction)
    one_hot[output] = 1
    delta = prediction - one_hot
    gradient_W = delta.dot(x.T)
    gradient_U = delta.dot(hidden_layer.T)
    gradient_H = tanh_gradient(hidden_layer) * U.T.dot(delta).dot(x.T)
    gradient_C = np.zeros_like(C)
    gradient_y_x = W + U.dot(tanh_gradient(hidden_layer) * H)
    gradient_x = gradient_y_x.T.dot(delta)
    gradient_x = gradient_x[:-1, :]

    gradient_x_split = np.split(gradient_x, context_size)
    for i in range(context_size):
        gradient_C[input[i]] += gradient_x_split[i].flatten()

    gradient = [gradient_W, gradient_U, gradient_H, gradient_C]
    return cost, gradient

