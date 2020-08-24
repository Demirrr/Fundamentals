import numpy as np
import copy
class Net:
    def __init__(self):
        self.gates = []

    def add(self, gate):
        self.gates.append(gate)

    def forward(self, inputs):
        for g in self.gates:
            inputs = g.forward(inputs)
        return inputs

    def backward(self, dL):
        for g in reversed(self.gates):
            dL = g.backward(dL)

    def update(self):
        for g in self.gates:
            g.update()

class SigmoidGate:
    def __init__(self, shape, learning_rate=.001):
        fout, fin = shape
        self.learning_rate = learning_rate
        self.W = np.random.randn(fout, fin) * .01
        self.b = np.ones((fout, 1)) * .01
        self.X = None
        self.S = None  # S=W.dot(x)+b
        self.Z = None  # Z=sigmoid(S)
        self.dLdW, self.dLdb, self.dLdX = None, None, None
        self.dZdS = None
        self.dSdW = None
        self.dSdX = None

    def update(self):
        self.W += -self.learning_rate * self.dLdW
        self.b += -self.learning_rate * self.dLdb

    def sigmoid(self, x):
        """
        Compute the sigmoid of x

        Arguments:
        x -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(x)
        """
        s = 1.0 / (1.0 + np.exp(-x))
        return s

    def dsigmoid(self, x):
        """
        Compute the derivative of sigmoid with respect to x

        Arguments:
        x -- A scalar or numpy array of any size.

        Return:
        ds -- (1-sigmoid(x)) * sigmoid(x)
        """
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)

    def forward(self, x):
        self.X = x
        self.S = self.W.dot(self.X) + self.b
        self.Z = self.sigmoid(self.S)
        # compute local gradients
        self.dZdS = self.dsigmoid(self.S)
        self.dSdW = self.X
        self.dSdX = self.W
        return self.Z

    def backward(self, dLdZ):
        assert self.Z.shape == dLdZ.shape
        # dL/dS= dZ/dS * dL/dZ
        dLdS = self.dZdS * dLdZ
        # dL/dW= dS/dW * dL/dS
        self.dLdW = dLdS.dot(self.dSdW.T)
        # dL/dX= dS/dX * dL/dS
        self.dLdX = self.dSdX.T.dot(dLdS)
        self.dLdb = np.sum(dLdS, axis=1, keepdims=True)
        return copy.deepcopy(dLdS)

class SoftmaxGate:
    def __init__(self, shape, learning_rate=.001):
        fout, fin = shape
        self.learning_rate = learning_rate
        self.W = np.random.randn(fout, fin) * 0.001
        self.b = np.ones((fout, 1)) * .001
        self.X, self.S = None, None
        self.dLdW, self.dLdb, self.dLdX = None, None, None
        self.dSdW, self.dSdX = None, None

    def update(self):
        self.W += -self.learning_rate * self.dLdW
        self.b += -self.learning_rate * self.dLdb

    def softmax(self, x, axis=0):
        """
            Vectorized computation of softmax function
            Adds a root node into the search tree.

            Parameters
            ----------
            x : shape=(N,K). s[i,j] represents the score of j.th class given i.th input.
            axis: rowise or column wise

            Returns
            -------
            shape=(N,K) [i,j] represents the predicted probability of j.th class given i.th input

        """
        x -= np.max(x, axis=axis, keepdims=True)
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)

    def forward(self, x):
        self.X = x
        self.S = self.W.dot(self.X) + self.b
        self.dSdW = self.X
        self.dSdX = self.W
        return self.softmax(self.S)

    def backward(self, dLdS):
        assert self.S.shape == dLdS.shape
        # Propagate dLdZ into dW,db, dS
        # dLdW= dS/dW * dL/dS
        self.dLdW = dLdS.dot(self.dSdW.T)
        assert self.dLdW.shape == self.W.shape

        # dLdX= dS/dX * dL/dS
        dLdX = self.dSdX.T.dot(dLdS)
        assert dLdX.shape == self.X.shape

        self.dLdb = np.sum(dLdS, axis=1, keepdims=True)
        return copy.deepcopy(dLdX)

class ReluGate:
    def __init__(self, shape, learning_rate=.001):
        fout, fin = shape
        self.learning_rate = learning_rate
        self.W = np.random.randn(fout, fin) * 0.001
        self.b = np.ones((fout, 1)) * .001
        self.S = None, None
        self.Z = None
        self.dLdW, self.dLdb = None, None
        self.dSdW, self.dSdX = None, None

    def update(self):
        self.W += -self.learning_rate * self.dLdW
        self.b += -self.learning_rate * self.dLdb

    def relu(self, X):
        return np.maximum(0, X)

    def forward(self, x):
        self.S = self.W.dot(x) + self.b
        self.dSdW = x
        self.dSdX = self.W
        self.Z = self.relu(self.S)
        return self.Z

    def backward(self, dLdZ):
        try:
            assert self.Z.shape == dLdZ.shape
        except:
            print(self.Z.shape)
            print(dLdZ.shape)
            exit(1)
        dLdZ[self.S <= 0] = 0
        dZdS = dLdZ
        # Propagate dLdZ into dW,db, dS
        # dLdW= dS/dW * dL/dS
        self.dLdW = dZdS.dot(self.dSdW.T)
        # dLdX= dS/dX * dL/dS
        dLdX = self.dSdX.T.dot(dZdS)
        self.dLdb = np.sum(dZdS, axis=1, keepdims=True)
        return copy.deepcopy(dLdX)