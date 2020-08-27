import numpy as np
import copy
from util import spiral_data_gen
from sklearn.metrics import classification_report
from sklearn.datasets import *
from nn import SoftmaxGate, ReluGate, Net

for X, y in [
    spiral_data_gen(False),
    (load_wine()['data'], load_wine()['target']),
    (load_breast_cancer()['data'], load_breast_cancer()['target']),
    (load_iris()['data'], load_iris()['target']),
    (load_digits()['data'], load_digits()['target']),
]:  # ,
    X -= np.mean(X, axis=0)  # zero-centerring.
    X = X.T
    D, N = X.shape
    K = len(np.unique(y))

    print(X.shape)

    hidden_size = 100
    model = Net()  # TODO weight decay impleement.
    model.add(ReluGate(shape=(hidden_size, D), learning_rate=.001))
    model.add(SoftmaxGate(shape=(K, hidden_size), learning_rate=.001))
    num_epoch = 10_000
    mode = num_epoch // 10

    for epoch in range(num_epoch):
        # forward
        f = model.forward(X)
        if epoch % mode == 0:
            loss = (-np.log(f[y, range(N)] + .01)).mean()  # compute the loss
            print('{0}.th epoch Loss:{1}'.format(epoch, loss))
            if loss < .001:
                break
        # backward
        dLdf = f
        dLdf[y, range(N)] -= 1
        dLdf /= N
        model.backward(dLdf)
        model.update()

    y_head = np.argmax(model.forward(X), axis=0)
    print(classification_report(y, y_head))
exit(1)


# TODO Understand gradient of softmax
# 2D inputs visualise the data and decision boundries.

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    return sm


def softmax_grad(s):
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x.
    # s.shape = (1, n)
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1 - s[i])
            else:
                jacobian_m[i][j] = -s[i] * s[j]
    return jacobian_m


def softmax_grad_vec(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)
