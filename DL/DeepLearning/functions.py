import numpy as np

import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1.0/(1.0+np.exp(-x))
    return s

def dsigmoid(x):
    """
    Compute the derivative of sigmoid with respect to x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    ds -- (1-sigmoid(x)) * sigmoid(x)
    """
    return (1.0-sigmoid(x)) * sigmoid(x)

def log2(p):
    return np.log2(p, out=np.zeros_like(p), where=(p!=0))

def log10(p):
    return np.log10(p, out=np.zeros_like(p), where=(p!=0))

def softmax(s,axis=0):
    """
        Vectorized computation of softmax function
        Adds a root node into the search tree.

        Parameters
        ----------
        s : shape=(N,K). s[i,j] represents the score of j.th class given i.th input.

        Returns
        -------
        probs:shape=(N,K) probs[i,j] represents the predicted probability of j.th class given i.th input
        
        Examples
        -------
    """
    s-=np.max(s,axis=axis,keepdims=True)
    exp_scores=np.exp(s)
    probs=exp_scores/np.sum(exp_scores,axis=axis,keepdims=True)
    return probs





def entropy(p):
    """ Vectorized Entropy - p is a N by D numpy array  - Return a N by 1 numpy array"""
    assert np.all(np.sum(p,axis=1)==1)    
    return -np.nansum(p*log2(p),axis=1)

def cross_entropy(p,q):
    """Vectorized Cross entropy - p, q true and predicted dist."""
    assert np.all(np.sum(p,axis=1)==1)
    assert np.all(np.sum(q,axis=1)==1)
    return -np.nansum(p*log2(q),axis=1)

def kl_div(p,q):
    assert np.all(np.sum(p,axis=1)==1)
    assert np.all(np.sum(q,axis=1)==1)
    return cross_entropy(p,q)-entropy(p)


"""


p=np.array([.0,1.0]).reshape(1,2)
print('Cross Entropy of probs.:',cross_entropy(p,p))
print('KL-divergence of probs:',cross_entropy(p,p))

q=np.array([.1,.9]).reshape(1,2)
print('Cross Entropy of probs.:',cross_entropy(p,q))
print('KL-divergence of probs:',kl_div(p,q))


q=np.array([.4,.6]).reshape(1,2)
print('Cross Entropy of probs.:',cross_entropy(p,q))
print('KL-divergence of probs:',kl_div(p,q))


p=np.eye(3)
q=np.array([[1.0,.0,.0],
            [.0,.8,.2],
            [.2,.3,.5]])

print('Cross Entropy of probs.:',cross_entropy(p,q))
print('KL-divergence of probs:',kl_div(p,q))
"""