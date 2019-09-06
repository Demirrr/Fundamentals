"""
This module wants to be a summary of the Deep NN course (1) on Coursera.
There will be a class to create a general L layers NN with variable number of activation units.
"""

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


class DeepNeuralNetwork(object):
    def __init__(self, layer_dimensions):
        """
        This class is used to create a deep neural network with L layers.
        Specify the number of layers, the number of nodes of each layer and the type of activation function via the
        `layer_dimension` parameter.

        Layer dimension must be a list of tuples containing (number of nodes, ActivationFunction).

        Attributes
        ----------

        layer_dimensions : list of tuples containing the layout of the DNN

        parameters : dictionary with the parameters (W,b) of the neural network

        :param layer_dimensions:
        """
        self.layer_dimensions = layer_dimensions
        self.__parameters = self.__initialize_deep_nn(self.layer_dimensions)
        self.__cache = {}
        self.__beta1 = None
        self.__beta2 = None
        self.__beta = None
        self.__iteration = None
        self.__cost = []
        self.__regularization = None
        self.__reg_lambda = 0

    @property
    def parameters(self):
        return self.__parameters

    def __initialize_deep_nn(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing a tuple with the dimensions of each layer in our network
        and the activation function

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        parameters = {}
        for l in range(1, len(layer_dims)):
            parameters["act_" + str(l)] = layer_dims[l][1]
            parameters["W" + str(l)] = np.random.randn(layer_dims[l][0], layer_dims[l - 1][0]) * 0.01
            parameters["b" + str(l)] = np.zeros((layer_dims[l][0], 1))
        parameters["L"] = len(layer_dims) - 1
        return parameters

    # Forward propagation step
    def __forward_block(self, A_previous, thisW, thisb, thisactivation):
        """
        Block to compute the forward propagation
        """
        Z = np.dot(thisW, A_previous) + thisb
        A = thisactivation.f(Z)
        return A, Z

    def forward_propagation(self, X):
        A = X
        L = (self.parameters["L"])
        self.__cache["A0"] = X
        for l in range(1, L + 1):
            A, Z = self.__forward_block(A, self.parameters["W" + str(l)], self.parameters["b" + str(l)],
                                        self.parameters["act_" + str(l)])
            self.__cache["A" + str(l)] = A
            self.__cache["Z" + str(l)] = Z
            self.__cache["VdW" + str(l)] = 0
            self.__cache["Vdb" + str(l)] = 0
            self.__cache["SdW" + str(l)] = 0
            self.__cache["Sdb" + str(l)] = 0
        return A

    # Backward propagation
    def __backward_block(self, dA, A_prev, thisW, thisZ, thisactivation):
        m = thisZ.shape[1]
        dZ = dA * thisactivation.grad_f(thisZ)
        dW = np.dot(dZ, A_prev.T) / m
        db = dZ.sum(axis=1, keepdims=True) / m
        dA_prev = np.dot(thisW.T, dZ)
        return dA_prev, dW, db

    def backward_propagation(self, Y):
        L = self.parameters["L"]
        AL = self.__cache["A" + str(L)]
        if self.parameters["act_" + str(self.parameters["L"])].activation_type is ActivationType.sigmoid:
            dA = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        else:
            raise NotImplementedError("Not yet implemented")

        gradients = {}
        for l in range(self.parameters["L"], 0, -1):
            dA, dW, db = self.__backward_block(dA, self.__cache["A" + str(l - 1)], self.parameters["W" + str(l)],
                                               self.__cache["Z" + str(l)], self.parameters["act_" + str(l)])
            gradients["dW" + str(l)] = dW
            gradients["db" + str(l)] = db
        return gradients

    def upgrade_weights(self, grads, learning_rate=0.05,
                        opt_scheme: OptimizationScheme = OptimizationScheme.gradientdescent):
        L = self.parameters["L"]
        for l in range(1, L + 1):
            if opt_scheme == OptimizationScheme.gradientdescent:
                correction_dW = grads["dW" + str(l)]
                correction_db = grads["db" + str(l)]
            elif opt_scheme == OptimizationScheme.momentum:
                self.__cache["VdW" + str(l)] = self.__beta * self.__cache["VdW" + str(l)] + (1 - self.__beta) * grads[
                    "dW" + str(l)]
                self.__cache["Vdb" + str(l)] = self.__beta * self.__cache["Vdb" + str(l)] + (1 - self.__beta) * grads[
                    "db" + str(l)]
                correction_dW = self.__cache["VdW" + str(l)]
                correction_db = self.__cache["Vdb" + str(l)]
            elif opt_scheme == OptimizationScheme.adam:
                self.__cache["VdW" + str(l)] = self.__beta1 * self.__cache["VdW" + str(l)] + (1 - self.__beta1) * grads[
                    "dW" + str(l)]
                self.__cache["Vdb" + str(l)] = self.__beta1 * self.__cache["Vdb" + str(l)] + (1 - self.__beta1) * grads[
                    "db" + str(l)]
                self.__cache["SdW" + str(l)] = self.__beta2 * self.__cache["SdW" + str(l)] + (1 - self.__beta2) * grads[
                    "dW" + str(l)] ** 2
                self.__cache["Sdb" + str(l)] = self.__beta2 * self.__cache["Sdb" + str(l)] + (1 - self.__beta2) * grads[
                    "db" + str(l)] ** 2

                # bias correction
                VdW = self.__cache["VdW" + str(l)] / (1 - self.__beta1 ** self.__iteration)
                Vdb = self.__cache["Vdb" + str(l)] / (1 - self.__beta1 ** self.__iteration)
                SdW = self.__cache["SdW" + str(l)] / (1 - self.__beta2 ** self.__iteration)
                Sdb = self.__cache["Sdb" + str(l)] / (1 - self.__beta2 ** self.__iteration)
                correction_dW = VdW / (np.sqrt(SdW) + 1e-8)
                correction_db = Vdb / (np.sqrt(Sdb) + 1e-8)
            else:
                correction_dW = grads["dW" + str(l)]
                correction_db = grads["db" + str(l)]

            # Add regularization contribution
            if self.__regularization == Regularization.ridge:
                reg_contr = self.parameters["W" + str(l)]
            elif self.__regularization == Regularization.lasso:
                reg_contr = np.sign(self.parameters["W" + str(l)])
            else:
                reg_contr = 0

            m = self.__cache["A0"].shape[1]

            self.__parameters["W" + str(l)] -= learning_rate * (correction_dW + self.__reg_lambda * reg_contr / m)
            self.__parameters["b" + str(l)] -= learning_rate * correction_db
        return self.parameters

    def compute_cost(self, Y, Yhat):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]
        AL = Yhat
        # Compute loss from aL and y.

        AL[AL >= 1] = 1 - np.finfo(float).eps
        AL[AL <= 0] = np.finfo(float).eps
        try:
            cost = -(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)).sum() / m
        except FloatingPointError:
            print(AL)

        # Add regularization
        if self.__regularization is not None:
            for l in range(1, self.parameters["L"] + 1):
                if self.__regularization == Regularization.ridge:
                    cost += self.__reg_lambda / m / 2 * (self.parameters["W" + str(l)] ** 2).sum()
                elif self.__regularization == Regularization.lasso:
                    cost += self.__reg_lambda / m * np.abs(self.parameters["W" + str(l)]).sum()

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())

        return cost

    def single_iteration(self, X, Y, learning_rate=0.05, minibatch_size=None,
                         opt_scheme: OptimizationScheme = OptimizationScheme.gradientdescent, dev_set=None):

        m = X.shape[1]
        if minibatch_size is not None:
            if minibatch_size >= m:
                minibatch_size = m
        perm = np.random.permutation(m)
        X_shuffled = X[:, perm]
        Y_shuffled = Y[:, perm]
        if minibatch_size is None:
            minibatch_size = m
        num_minibatch_iteration = int(np.floor(m / minibatch_size))

        nn_parameters, cost = None, None

        for miniter in range(num_minibatch_iteration):
            self.__iteration += 1
            k0 = miniter * minibatch_size
            k1 = (miniter + 1) * minibatch_size
            yhat = self.forward_propagation(X_shuffled[:, k0:k1])
            cost = self.compute_cost(Y_shuffled[:, k0:k1], yhat)
            gradients = self.backward_propagation(Y_shuffled[:, k0:k1])
            nn_parameters = self.upgrade_weights(gradients, learning_rate, opt_scheme)
            if dev_set is not None:
                yhat = self.predict(dev_set[0])
                cost_dev = self.compute_cost(dev_set[1], yhat)
            else:
                cost_dev = np.nan
            self.__cost.append([cost, cost_dev])

        if m % minibatch_size != 0:
            self.__iteration += 1
            yhat = self.forward_propagation(X_shuffled[:, k1:])
            cost = self.compute_cost(Y_shuffled[:, k1:], yhat)
            gradients = self.backward_propagation(Y_shuffled[:, k1:])
            nn_parameters = self.upgrade_weights(gradients, learning_rate, opt_scheme)
            if dev_set is not None:
                yhat = self.predict(dev_set[0])
                cost_dev = self.compute_cost(dev_set[1], yhat)
            else:
                cost_dev = np.nan
            self.__cost.append([cost, cost_dev])

        return nn_parameters, cost

    def gradient_descent(self, X, Y, learning_rate=0.5, num_iter=3000,
                         minibatch_size=None,
                         optimization_scheme: OptimizationScheme = OptimizationScheme.gradientdescent,
                         regularization: Regularization = None,
                         **kwargs):

        dev_set = kwargs.get("dev_set", None)

        self.__iteration = 0
        self.__regularization = regularization
        print(optimization_scheme.value)

        if optimization_scheme == OptimizationScheme.momentum:
            self.__beta = kwargs.get("beta", 0.9)
        elif optimization_scheme == OptimizationScheme.adam:
            self.__beta1 = kwargs.get("beta1", 0.9)
            self.__beta2 = kwargs.get("beta2", 0.999)

        if regularization is not None:
            self.__reg_lambda = kwargs.get("lambd")
        for i in range(num_iter):
            nn_parameters, cost = self.single_iteration(X, Y, learning_rate, minibatch_size, optimization_scheme,
                                                        dev_set)
            if i % 500 == 0 and kwargs.get("verbose", True):
                print("Iteration {0}; Cost {1}".format(i, cost))

        return self.__cost

    def predict(self, X, pred_type="binary"):
        ypred = self.forward_propagation(X)
        if pred_type == "binary":
            return np.round(ypred)

    def plot_learning_curve(self, fig=None, title="", logarithmic=True):
        if fig is None:
            fig = plt.figure()
        else:
            plt.figure(num=fig.number)
        cost = np.array(self.__cost)
        iternum = np.arange(cost.shape[0])

        plt.plot(iternum, cost[:, 0], label="Training error")
        if cost.shape[1] == 2:
            plt.plot(iternum, cost[:, 1], label="Dev error")
        plt.legend()
        plt.title(title)
        if logarithmic:
            plt.yscale("log")


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler

    import matplotlib.pyplot as plt

    data = load_breast_cancer()
    X = data["data"]
    y = data["target"]

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    nn_structure = [(X_train.shape[0], ActivationFunction(activation_type=ActivationType.leakyrelu)),
                    (30, ActivationFunction(activation_type=ActivationType.leakyrelu)),
                    (30, ActivationFunction(activation_type=ActivationType.leakyrelu)),
                    (1, ActivationFunction(activation_type=ActivationType.sigmoid))]

    # Same learning rate, different optimizations
    for reg in [Regularization.lasso]:
        for lambd in np.logspace(-5, -3, 5):
            thisNN = DeepNeuralNetwork(nn_structure)
            cost = thisNN.gradient_descent(X_train, y_train, learning_rate=0.00001, num_iter=20000,
                                           minibatch_size=None,
                                           optimization_scheme=OptimizationScheme.adam,
                                           regularization=reg,
                                           lambd=lambd,
                                           dev_set=[X_test, y_test])
            thisNN.plot_learning_curve(title="{0}, lambda: {1}".format(reg.value, lambd), logarithmic=True)
            ypred = thisNN.predict(X_test)
            print(classification_report(y_test.T, ypred.T))
    plt.show()