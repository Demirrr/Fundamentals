import numpy as np

from sklearn.metrics import classification_report

from functions import softmax



def SoftmaxClassifer(X,y,num_epoch=50_000,l2_reg=.0,step_size=.01):
    """
    X is of N \times D where N =>number of data points, D is num of dim.
    y is a numpy array of (N \times 1), (N)
    """
    assert len(X)==len(y)
    N,D=X.shape
    K=len(np.unique(y))
    X=X.T
    X=np.vstack((X,np.ones(N)))# add 1's as rows indicating a new dim for bias
    W=np.random.randn(K,D+1) * 0.0001 # generate random parameters

    mode=num_epoch//10
    print('Softmax training starts.')
    #################### Training starts #####################
    for epoch in range(num_epoch):

        # Compute predictions
        Z = softmax(W.dot(X),axis=0)
        # Compute loss
        correct_logprobs = (-np.log( Z[y,range(N)]+.0001))+.01
        data_loss = np.sum(correct_logprobs)/N

        reg_loss = 0.5*l2_reg*np.sum(W*W) # multiplying reg with .5 simplifies gradient of reg.
        loss = data_loss + reg_loss
        
        if epoch % mode == 0:
            print ("Epoch {0}: Cost:{1}".format(epoch, loss))    
        
        # Compute Gradients and backpropagate them
        dZ = Z
        # 1 - predicted probability of correct classes.
        dZ[y,range(N)] -= 1 
        # backpropate the gradient to the parameters (W)    
        dW = dZ.dot(X.T)
        
        dW += l2_reg*W # regularization gradient
            
        # perform a parameter update
        W += -step_size * dW
    #################### Training ends #####################
    predicted_class = np.argmax(softmax(W.dot(X)), axis=0)
    print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
    print(classification_report(y, predicted_class))
    return W


def MaximumMarginClassifier(X,y,num_epoch=50_000,l2_reg=.0,step_size=.01):
    """
    X is of N \times D where N =>number of data points, D is num of dim.
    y is a numpy array of (N \times 1), (N)
    """
    assert len(X)==len(y)

    N,D=X.shape
    K=len(np.unique(y))
    X=X.T
    X=np.vstack((X,np.ones(N)))# add 1's as rows indicating a new dim for bias
    W=np.random.randn(K,D+1) * 0.0001 # generate random parameters
        
    mode=num_epoch//10
    
    print('MaximumMarginClassifier training starts')
    #################### Training starts #####################
    for epoch in range(num_epoch):
        
        # Compute Scores
        Z =W.dot(X)   
        y_true_scores=Z[y,range(len(y))]
        # Compute margin for all classes
        margins=np.maximum(0,Z-y_true_scores+1)
        margins[y,range(len(y))]=0
        loss=margins.sum()
        
        # Compute the derivative of loss w.r.t. weights.
        binary=margins
        binary[margins > 0] = 1
    
        
        sum_of_error= np.sum(binary, axis=0)
        binary[y,range(len(y))] = -sum_of_error.T
        dW = binary.dot(X.T)

        
        # perform a parameter update
        W += -step_size * dW
    
    #################### Training ends #####################
    predicted_class = np.argmax(W.dot(X), axis=0)
    print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
    print(classification_report(y, predicted_class))
    return W
    