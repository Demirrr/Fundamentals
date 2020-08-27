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


class Gate:  # Abstract.
    def __init__(self, shape, learning_rate=.001, xavier_init=False):
        fout, fin = shape
        self.learning_rate = learning_rate
        self.W = np.random.randn(fout, fin)
        self.b = np.ones((fout, 1))
        if xavier_init:
            self.W /= np.sqrt(fin)
        else:
            self.W *= .001
            self.b *= .001
        self.X = None
        self.S = None  # S=W.dot(x)+b
        self.Z = None  # activation
        self.dLdW, self.dLdb, self.dLdX = None, None, None
        self.dZdS = None
        self.dSdW = None
        self.dSdX = None

        self.decay_rate = .99
        self.cache_w = np.zeros(self.W.shape)
        self.cache_b = np.zeros(self.b.shape)

    def update(self):
        # TODO imlement weight decay.

        # Gradient Descent
        # self.W += -self.learning_rate * self.dLdW
        # self.b += -self.learning_rate * self.dLdb

        # AdaGrad update
        # self.cache_w += self.dLdW ** 2
        # self.cache_b += self.dLdb ** 2
        # self.W += -self.learning_rate * self.dLdW / (np.sqrt(self.cache_w) + 1e-7)
        # self.b += -self.learning_rate * self.dLdb / (np.sqrt(self.cache_b) + 1e-7)

        # RMSProb
        self.cache_w = self.decay_rate * self.cache_w + (1 - self.decay_rate) * self.dLdW ** 2
        self.cache_b = self.decay_rate * self.cache_b + (1 - self.decay_rate) * self.dLdb ** 2
        self.W += -self.learning_rate * self.dLdW / (np.sqrt(self.cache_w) + 1e-7)
        self.b += -self.learning_rate * self.dLdb / (np.sqrt(self.cache_b) + 1e-7)

        # ADAM
        # m =  beta1 * m + (1-beta1) * dx # Momentum
        # v = beta2 * v + (1-beta2) * (dx**2) # RMSPROB like
        # m /= 1-beta1**t
        # v /= 1 - beta2 ** t
        # self.W += -self.learning_rate * m / (np.sqrt(v) + 1e-7) # RMSPROB like


class SigmoidGate(Gate):
    def __init__(self, shape, learning_rate, xavier_init=False):
        super().__init__(shape, learning_rate, xavier_init)

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


class SoftmaxGate(Gate):
    def __init__(self, shape, learning_rate, xavier_init=False):
        super().__init__(shape, learning_rate, xavier_init)

    """
    
    def __init__(self, shape, learning_rate=.001,xavier_init=False):
        fout, fin = shape
        self.learning_rate = learning_rate
        self.W = np.random.randn(fout, fin)
        self.b = np.ones((fout, 1))
        if xavier_init:
            self.W /= np.sqrt(fin)
        else:
            self.W *= .001
            self.b *= .001

        self.X, self.S = None, None
        self.dLdW, self.dLdb, self.dLdX = None, None, None
        self.dSdW, self.dSdX = None, None
    """

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


class ReluGate(Gate):
    def __init__(self, shape, learning_rate, xavier_init=False):
        super().__init__(shape, learning_rate, xavier_init)

        if xavier_init:
            self.W /= np.sqrt(fin / 2)  # He et al. 2015
        else:
            self.W *= .001
            self.b *= .001

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


class ELUGate:
    pass


class Maxout:
    pass


class BatchNorm:
    def __init__(self, ):
        # add after fully connected layers or before nonlinearity.
        pass

    def forward(self, X):
        # normalize X.
        # y=gamma * normalized X + Beta
        # return y

        pass

    def backward(self):
        pass
