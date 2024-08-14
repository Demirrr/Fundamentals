from abc import ABC, abstractmethod
import numpy as np

class Gate(ABC):
    def __init__(self):
        self.num_param=0           
        self.weight, self.bias=None, None
        self.dweight, self.dbias=None, None
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    @abstractmethod
    def backward(self, *args, **kwargs):
        pass
    @property
    def param_size(self):
        if self.weight is None and self.bias is None:
            return 0
        return self.weight.size+self.bias.size   
    def dim_checker(self,a,b):
        return a.shape==b.shape
    
class ResidualBlock(Gate):
    def __init__(self,in_features=None,out_features=None,res_flow=None,identity_flow=None):
        super().__init__()
        if res_flow is None :
            self.a=[Linear(in_features, in_features), 
                    BatchNorm(in_features),
                    ReLU(),
                    Linear(in_features, out_features)]
        else:
            self.a=res_flow
        
        if identity_flow is None:
            self.b=[Linear(in_features, out_features)]    
        else:
            self.b=identity_flow
    @property
    def param_size(self):
        return sum([g.param_size for g in self.a+self.b])
    
    def __str__(self):
        return f'ResidualBlock {self.param_size}'
    
    def __iter__(self):
        for i in self.a+self.b:
            yield i
        
    def forward(self, X):
        """
        Parameters
        ----------
        X : shape=(N,in_features)
        Returns
        ----------
        Z:shape(N,out_features)
        """
        res_a= np.zeros(X.shape)+X 
        res_b= np.zeros(X.shape)+X
        for g in self.a:
            res_a=g.forward(res_a)
        for g in self.b:
            res_b=g.forward(res_b)
        return res_a+res_b
        
    def backward(self, dZ):
        """
        Parameters
        ----------
        dZ : shape=(N,out_features)
        ----------
        Returns
        -------
        dX : shape (N,in_features)
        """
        res_a  = np.ones(dZ.shape) *dZ
        res_b  = np.ones(dZ.shape) *dZ
        
        for g in reversed(self.a):
            res_a= g.backward(res_a)
        for g in reversed(self.b):
            res_b= g.backward(res_b)
        return res_a+res_b
    
class ReLU(Gate):
    def __init__(self,prob=.1):
        super().__init__()
        self.cache=dict()
        
    def __str__(self):
        return f'ReLU'
    
    def forward(self, X):
        """
        X
        ----------
        X : shape=(N,D) or         X : shape=(N, K, D)
        Returns
        -------
        Z : shape=(N,D) or         X : shape=(N, K, D)

        """
        self.cache['X']=X
        Z=np.maximum(0,X)
        return Z
    def backward(self, dZ):
        dZ[self.cache['X'] <= 0] = 0
        return dZ
    
class BatchNorm(Gate):
    """
    Batch normalization.
    """
    def __init__(self,dim,eps=.001):
        super().__init__()
        self.eps=eps
        self.weight=np.random.randn(dim) # scale
        self.bias=np.random.randn(dim) # shift
        self.num_param=self.weight.size+ self.bias.size
        
        self.cache=None
        # Backward
        self.dweight=None
        self.dbias=None
        
    def __str__(self):
        return f'Batch Norm.:epsilon:{self.eps}, Param.{self.param_size}'
    
    def forward(self, X):
        """
        X
        ----------
        X : shape=(N,D)

        Returns
        -------
        Z : shape (N,D)
        """
        
        N,D=X.shape
        # 1. Compute the mini-batch mean.
        mu=X.mean(axis=0)  
        
        # 2. Compute the mini-batch variance -X.var(axis=0)-.
        # 2.1
        xmu = X - mu
        # 2.2
        sq = xmu ** 2
        var = 1./N * np.sum(sq, axis = 0)
        
        #3. Compute the denominator of normalization
        sqrtvar = np.sqrt(var + self.eps)
        #3.1 invert sqrtwar
        ivar = 1./sqrtvar
        
        #4. Execute normalization
        xhat = xmu * ivar
        
        #5. Nor the two transformation steps
        gammax = self.weight*xhat#self.gamma * xhat
        
        #6.
        Z = gammax + self.bias#gammax + self.beta
        #store intermediate
        self.cache = (xhat,xmu,ivar,sqrtvar,var)        
        return Z

    def backward(self, dZ):
        N, D = dZ.shape
        xhat,xmu,ivar,sqrtvar,var=self.cache
        
        #6. 
        ## Forward: gammax + self.beta = Z 
        ## Backward: dgammax= Z and self.dbeta= Z
        self.dbias = np.sum(dZ, axis=0); assert self.dbias.shape==self.bias.shape #Sanity checking
        dgammax = dZ

        #5.
        ## Forward: self.gamma * xhat = gammax
        ## Backward: self.dgamma = dgammax * xhat and dxhat= dgammax * self.gamma
        self.dweight = np.sum(dgammax*xhat, axis=0); assert self.dweight.shape==self.weight.shape
        dxhat = dgammax * self.weight

        #4.
        divar = np.sum(dxhat*xmu, axis=0)
        dxmu1 = dxhat * ivar

        #3.(derivative of quotient) and derivative of root.
        dsqrtvar = -1. /(sqrtvar**2) * divar
        dvar = 0.5 * 1. /np.sqrt(var+self.eps) * dsqrtvar

        #2.
        dsq = 1. /N * np.ones((N,D)) * dvar

        #step3
        dxmu2 = 2 * xmu * dsq

        #step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

        #step1
        dx2 = 1. /N * np.ones((N,D)) * dmu

        #step0
        dx = dx1 + dx2

        return dx
    
class Conv(Gate):
    def __init__(self,in_channels=1, out_channels=1,kernel_size=(2, 2), stride=1, padding=0):        
        super().__init__()
        self.kernel_h,self.kernel_w=kernel_size
        self.weight=np.random.randn(out_channels,
                               in_channels,
                               self.kernel_h,
                               self.kernel_w) /np.sqrt(in_channels/2)
        self.bias=np.zeros(out_channels)    
        self.stride=stride
        self.padding=padding
        self.cache=dict()

    def __str__(self):
        return f'Conv: (out_channels, in_channels, h, w):{self.weight.shape}. Stride:{self.stride}. Params. {self.param_size}: '

    def set_params(self,weights,bias=None):
        self.weight,self.bias=weights, bias
        n,d,self.kernel_h,self.kernel_w=self.weight.shape        

    def compute_dim(self,X):
        # parameter check
        xN, xD, xH, xW = X.shape
        wN, wD, wH, wW = self.weight.shape
        assert wH == wW
        assert (xH - wH) % self.stride == 0
        assert (xW - wW) % self.stride == 0
        self.cache['X']=X
        
        zH, zW = (xH - wH) // self.stride + 1, (xW - wW) // self.stride + 1
        zD,zN = wN,xN
        return np.zeros((zN, zD, zH, zW))
    
    def get_region(self,hight,width):
        h1=hight*self.stride
        h2=h1+self.kernel_h
        w1=width*self.stride
        w2=w1+self.kernel_w
        return h1,h2,w1,w2
    
    def convolve_forward_step(self,X_n):
        xD, xH, xW = X_n.shape
        hZ=int((xH-self.kernel_h)/self.stride+1)
        wZ=int((xW-self.kernel_w)/self.stride+1)
        Z = np.zeros((len(self.weight),hZ, wZ))
        
        for d in range(len(Z)):
            for i in range(hZ):
                for j in range(wZ):
                    h1,h2,w1,w2=self.get_region(i,j)
                    x_loc = X_n[:, 
                              h1: h2,
                              w1: w2]
                    Z[d,i,j]=np.sum(x_loc*self.weight[d])+ self.bias[d]
        return Z
    
    def forward(self,X):
        Z=self.compute_dim(X)
        for n in range(len(Z)):
            Z[n,:,:,:]=self.convolve_forward_step(X[n])
        self.cache['Z']=Z
        return Z
    
    def backward(self,dZ):        
        assert self.dim_checker(dZ,self.cache['Z'])
        
        dX, self.dweight, self.dbias=np.zeros(self.cache['X'].shape), np.zeros(self.weight.shape),np.zeros(self.bias.shape)
        (N, depth, hight, width) = dZ.shape
         
        for n in range(N):
            for h in range(hight):        
                for w in range(width):      
                    for d in range(depth): # correcponds to d.th kernel
                        h1,h2,w1,w2=self.get_region(h,w)
                        dX[n,:,h1:h2,w1:w2]+= self.weight[d,:,:,:] * dZ[n, d, h, w]
                        self.dweight[d,:,:,:] += self.cache['X'][n, :, h1:h2, w1:w2] * dZ[n, d, h, w]            
                        self.dbias[d] +=dZ[n, d, h, w]
                    
        return dX
    
class Linear(Gate):
    def __init__(self,in_features, out_features):
        super().__init__()
        self.weight=np.random.rand(in_features,out_features)/in_features
        self.bias=np.random.rand(out_features)
        self.cache=dict()
    
    def __str__(self):
        return f'Linear: W:{self.weight.shape}:Params. {self.param_size}'

        
    def forward(self, X):
        """
        Parameters
        ----------
        X : shape=(N,in_features)
        
        Returns
        ----------
        Z:shape(N,out_features)
        """
        self.cache['X']=X
        Z=X.dot(self.weight)+self.bias
        return Z
    
    def backward(self, dZ):
        """
        Parameters
        ----------
        dZ : shape=(N,out_features)
        
        ----------
        dX: dZ (N,out_features) * weight (out_features,in_features) => (N,in_features)

        dW: X.T (in_features,N,) * dZ (N,out_features) => (in_features,out_features)
        
        db: dZ= (out_features,)

        Returns
        -------
        dX : shape (N,in_features)
        """
        dX= dZ.dot(self.weight.T) # dL/dZ * dZ/dX
        self.dweight= self.cache['X'].T.dot(dZ)        # dZ/dX* dL/dZ
        self.dbias=dZ.sum(axis=0)
        try:
            assert self.dweight.shape==self.weight.shape
            assert self.dbias.shape==self.bias.shape
        except AssertionError as a:
            print(self.weight.shape)
            print(self.dweight.shape)

            print(self.bias.shape)
            print(self.dbias.shape)
            
            raise
        return dX
    
    
    
class Reshape(Gate):
    def __init__(self,out_shape=None,flatten=False):
        super().__init__()
        self.flatten=flatten
        self.out_shape=out_shape     
        self.cache=dict()
    def __str__(self):
        return f'Reshape: Flatten:{self.flatten}, OutShape: {self.out_shape}'

    def forward(self, X):
        self.cache['X']=X
        if self.flatten==True:
            self.out_shape=(len(X), X.size//len(X))
        Z= np.reshape(X,self.out_shape)       
        return Z
    def backward(self, dL_dZ):
        dX= dL_dZ.reshape(self.cache['X'].shape)        
        return dX
    
class Softmax(Gate):
    def __init__(self):
        super().__init__()
    def __str__(self):
        return f'Softmax'

    def forward(self, X,axis=1):
        assert len(X.shape)==2
        X-=np.max(X,axis=axis,keepdims=True)
        exp_scores=np.exp(X)
        Z=exp_scores/np.sum(exp_scores,axis=axis,keepdims=True)
        return Z
    def backward(self, dL_dZ):
        return dL_dZ


X=np.random.rand(3,5)
Z=Softmax().forward(X)
assert Z.shape==X.shape
assert Z[0,:].sum()>.9999
assert Z[1,:].sum()>.9999
assert Z[2,:].sum()>.9999
del X,Z
