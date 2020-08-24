import numpy as np
from sklearn.datasets import *
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
np.random.seed(0)
def softmax(s):
    """ s is of D x N and probs is of D x N """
    #assert s.shape[0]>1
    s-=np.max(s,axis=0,keepdims=True)
    exp_scores=np.exp(s)
    probs=exp_scores/np.sum(exp_scores,axis=0,keepdims=True)
    return probs

def SoftmaxClassifer(X,y,num_epoch=1000,l2_reg=0,step_size=.01):
    """
    X \in R^{N \times D} where N denotes the number of data points and D denotes dim of input space.
    y \in R^{N}
    """
    N,D=X.shape
    X=X.T
    K=len(np.unique(y))
    # initialize parameters randomly
    W = 0.01 * np.random.randn(K,D)
    b = np.zeros((K,1))
    
    mode=num_epoch//10

    # gradient descent loop
    num_examples = X.shape[0]
    for i in range(num_epoch):
        # compute the class probabilities [K x N]
        probs=softmax(W.dot(X)+b)
        
        # compute the loss: average cross-entropy loss and regularization
        correct_logprobs = -np.log(probs[y,range(N)])

        data_loss = np.sum(correct_logprobs)/N
        reg_loss = 0.5*l2_reg*np.sum(W*W) # multiplying reg with .5 simplifies gradient of reg.
        loss = data_loss + reg_loss
        if i % mode == 0:
            print ("iteration %d: loss %f" % (i, loss))
        # compute the gradient on scores
        dscores = probs
        dscores[y,range(N)] -= 1
        dscores /= num_examples
        

        # backpropate the gradient to the parameters (W,b)
        dW = dscores.dot(X.T)
        db = np.sum(dscores, axis=1, keepdims=True)

        dW += l2_reg*W # regularization gradient

        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        

    probs = softmax(W.dot(X) + b)
    predicted_class = np.argmax(probs, axis=0)
    print(classification_report(y, predicted_class))
    W=W.T # Due to visualization
    b=b.T
    return W,b

#X,y=spiral_data_gen(True)
#W,b=SoftmaxClassifer(X,y)
#plot_decision_boundries(X,y,W,b)

def NNClassifier(X,y,num_epoch=100,reg=1e-3,step_size=1e-1):
    """ X \in R^{N \times D}
        y \in Natural Number^{N} where R is real valu, N
    """
    X=X.T
    D,N=X.shape
    K=len(np.unique(y))
    
    # initialize parameters randomly
    h = 100 # size of hidden layer
    W = 0.01 * np.random.randn(h,D)
    b = np.zeros((h,1))
    W2 = 0.01 * np.random.randn(K,h)
    b2 = np.zeros((K,1))
    
    mode=num_epoch//10

    
    for i in range(num_epoch):
    
        # forward pass - compute predictions
        Z = np.maximum(0, W.dot(X)+ b)     
        S =softmax(W2.dot(Z) + b2)

        # compute the loss: average cross-entropy loss and regularization
        corect_logprobs = -np.log(S[y,range(N)])
        data_loss = np.sum(corect_logprobs)/N
        reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
        loss = data_loss + reg_loss
        
        if i % mode == 0:
            print("iteration %d: loss %f" % (i, loss))


        # compute the gradient on predicted probs
        # Derivative of loss w.r.t. ouput of softmax.
        dS = S
        dS[y,range(N)] -= 1
        dS /= N

        # backpropate the gradient to the parameters
        # first backprop into parameters W2 and b2  
        dW2=dS.dot(Z.T)
        db2 = np.sum(dS, axis=1, keepdims=True)


        dZ=W2.T.dot(dS) 
        # backprop the ReLU non-linearity
        dZ[Z <= 0] = 0

        # finally into W,b

        dW=dZ.dot(X.T)
        db = np.sum(dZ, axis=1, keepdims=True)


        # add regularization gradient contribution
        dW2 += reg * W2
        dW += reg * W

        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2
    
 
    probs=softmax(W2.dot(np.maximum(0, W.dot(X)+ b)) + b2)    
    predicted_class = np.argmax(probs, axis=0)
    print(classification_report(y, predicted_class))

    
    
    X=X.T
    W=W.T
    W2=W2.T
    b2=b2.T
    b=b.T

    # plot the resulting classifier
    #h = 0.02
    #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    #Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
    #Z = np.argmax(Z, axis=1)
    #Z = Z.reshape(xx.shape)
    #fig = plt.figure()
    #plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    #plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    #plt.xlim(xx.min(), xx.max())
    #plt.ylim(yy.min(), yy.max())
    
# X,y=spiral_data_gen(False)
#NNClassifier(X,y,num_epoch=10000,step_size=1.0)




class MatMulGate:
    def __init__(self):
        pass
    def forward(self,W,b,X):
        self.W=W
        self.X=X
        return self.W.dot(self.X)+b
    def backward(self,dL):
        dW=dL.dot(self.X.T)
        dX=self.W.T.dot(dL)
        db=np.sum(dL, axis=1, keepdims=True)
        return dW,dX,db

class ReluGate:
    def __init__(self):
        pass
    def forward(self,X):
        self.X=X
        return np.maximum(0, self.X)
    def backward(self,dL):
        dL[self.X <= 0] = 0
        return dL
  
class ELU:
    pass

class Maxout:
    pass

#IDEA
# Combine complex numbers in maxout.
class CMax:
    pass

class SoftmaxGate:
    def __init__(self):
        pass
    def forward(self,X):
        self.Z=softmax(X)
        return self.Z
    def backward(self,dL):
        return dL
    
class ComputationalGraph:
    def __init__(self,learning_rate=.001):
        self.gates=[]
        self.learning_rate=learning_rate
    def add(self,shape,gate,xavier_init=False):
        fout,fin=shape
        if xavier_init:
            W = np.random.randn(fout,fin) / np.sqrt(fout)
            b = np.ones((fout,1))
        else:
            W = 0.01 * np.random.randn(fout,fin)
            b = np.ones((fout,1))*.001

        self.gates.append(((W,b), MatMulGate(),gate))
    def forward(self,X):
        for t in self.gates:
            (W,b),score,sigma=t
            S=score.forward(W,b,X)
            X=sigma.forward(S)
        self.Z=X
        return self.Z
    
    def backward(self,y):
        # Derivative of loss w.r.t. S2 the input of of softmax.
        dL = self.Z
        dL[y,range(len(y))] -= 1
        for t in reversed(self.gates):
            (W,b),score,sigma=t
            dW,dL,db=score.backward(sigma.backward(dL))
            W+=-self.learning_rate*dW
            b+=-self.learning_rate*db