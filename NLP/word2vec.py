import numpy as np
from utilities import softmax


def softmax_cost_gradient(parameters, input, output):
    """
    Softmax cost and gradient function for word2vec models
    :param parameters: word vectors for input and output (shape: (2, vocabulary_size, vector_size))
    :param input: index to input word vectors
    :param output: index to output word vectors
    :return: cross entropy cost and gradient
    """
    input_vectors, output_vectors = parameters
    input_vector = input_vectors[input]
    prediction = softmax(output_vectors.dot(input_vector))

    one_hot_vector = np.zeros_like(prediction)
    one_hot_vector[output] = 1

    gradient_input = np.zeros_like(input_vectors)
    gradient_input[input] = output_vectors.T.dot(prediction - one_hot_vector)
    gradient_output = (prediction - one_hot_vector).reshape(-1, 1).dot(input_vector.reshape(-1, 1).T)
    gradient = np.array([gradient_input, gradient_output])

    cost = -np.log(prediction[output])
    return cost, gradient


def skip_gram_cost_gradient(parameters, input, outputs):
    """
    Skip-gram model in word2vec
    :param parameters: word vectors for input and output (shape: (2, vocabulary_size, vector_size))
    :param input: index of center word to input word vectors
    :param outputs: indices of context words to output word vectors (shape: (context_size))
    :return: cost and gradient
    """
    total_cost = 0.0
    total_gradient = np.zeros_like(parameters)
    for output in outputs:
        cost, gradient = softmax_cost_gradient(parameters, input, output)
        total_cost += cost
        total_gradient += gradient
    return total_cost, total_gradient


def create_context(sentences, context_size):
    """
    Extract pairs of context and center word for Skip-gram training
    :param sentences: list of sentence (list of words)
    :param context_size: integer of context size
    :return: inputs and outputs vectors
    """
    inputs = []
    outputs = []

    for sentence in sentences:
        for i in range(len(sentence)):
            output_row = []
            input = sentence[i]

            for j in range(-context_size, context_size + 1):
                if j == 0 or i + j < 0 or i + j >= len(sentence):
                    continue

                output = sentence[i + j]
                output_row.append(output)

            inputs.append(input)
            outputs.append(output_row)

    inputs = np.array(inputs).T
    outputs = np.array(outputs).T
    return inputs, outputs
