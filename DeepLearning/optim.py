from abc import ABC, abstractmethod
import numpy as  np
class Optim(ABC):
    def __init__(self,learning_rate=.001):
        self.learning_rate=learning_rate
    @abstractmethod
    def update(self, *args, **kwargs):
        pass
    
class ADAM(Optim):
    def __init__(self,learning_rate=.001):
        super().__init__()
        self.learning_rate=learning_rate
        self.cache = dict()
        self.beta1 = 0.9
        self.beta2=.999
        
        self.t=1 # counter
        
    def init_cache(self,weight_shape,bias_shape):
        return (np.zeros(weight_shape),np.zeros(weight_shape),
                np.zeros(bias_shape),np.zeros(bias_shape))
    
    def update_weight(self,g):
        m,v,_,_=self.cache[g]
        
        m=self.beta1 * m + (1 - self.beta1) * g.dweight
        mt=m/(1-self.beta1**self.t)
        
        v=self.beta2 * v + (1 - self.beta2) * (g.dweight**2)
        vt=v/(1-self.beta2**self.t)
    
        g.weight += -self.learning_rate * mt / (np.sqrt(vt) + 1e-8)
        self.cache[g]=m,v,_,_

    def update_bias(self,g):
        _,_,m,v=self.cache[g]
        
        m=self.beta1 * m + (1 - self.beta1) * g.dbias
        mt=m/(1-self.beta1**self.t)
        
        v=self.beta2 * v + (1 - self.beta2) * (g.dbias**2)
        vt=v/(1-self.beta2**self.t)
        
        
        g.bias += -self.learning_rate * mt / (np.sqrt(vt) + 1e-8)
        self.cache[g]=_,_,m,v
        
    def update(self,gates):
        for g in gates:
            if g.param_size==0:
                continue    
                
            try:
                
                if not (g in self.cache):
                    self.cache[g]=self.init_cache(g.weight.shape,g.bias.shape)
            except AttributeError as e:
                print(g)
                print(g.weight)
                raise 

            self.update_weight(g)
            self.update_bias(g)
            self.t+=1
class SGD(Optim):
    def __init__(self,learning_rate=.001):
        super().__init__()
        self.learning_rate=learning_rate
        
    def update(self,gates):
        for g in gates:
            if g.param_size>0:
                try:
                    g.weight += - self.learning_rate * g.dweight
                    g.bias   += - self.learning_rate * g.dbias
                except TypeError as e:
                    print(g)
                    print(g.weight)
                    print(g.bias)
                    print(g.dweight)
                    print(g.dbias)
                    raise 
class RMSProb(Optim):
    def __init__(self,learning_rate=.001):
        super().__init__()
        self.learning_rate=learning_rate
        self.decay_rate = .99
        self.cache = dict()        
    def update(self,gates):
        for g in gates:
            if g.param_size==0:
                continue
            if not (g in self.cache):
                self.cache[g]=(np.zeros(g.weight.shape),np.zeros(g.bias.shape))

            cache_weight,cache_bias=self.cache[g]

            cache_weight = self.decay_rate * cache_weight + (1 - self.decay_rate) * g.dweight ** 2
            cache_bias = self.decay_rate * cache_bias + (1 - self.decay_rate) * g.dbias ** 2

            g.weight += - self.learning_rate * g.dweight / (np.sqrt(cache_weight) + 1e-7)
            g.bias   += - self.learning_rate * g.dbias / (np.sqrt(cache_bias) + 1e-7)

            self.cache[g]=(cache_weight,cache_bias)
                    
class AdaGrad(Optim):
    def __init__(self,learning_rate=.001):
        super().__init__()
        self.learning_rate=learning_rate
        self.cache = dict()  
    
    def update(self,gates):
        for g in gates:
            if g.param_size==0:
                continue
                
            if not (g in self.cache):
                self.cache[g]=(np.zeros(g.weight.shape),np.zeros(g.bias.shape))
            cache_weight,cache_bias=self.cache[g]
            
            cache_weight += g.dweight ** 2
            cache_bias += g.dbias ** 2

            g.weight += - self.learning_rate * g.dweight / (np.sqrt(cache_weight) + 1e-7)
            g.bias   += - self.learning_rate * g.dbias / (np.sqrt(cache_bias) + 1e-7)

            self.cache[g]=(cache_weight,cache_bias)
            
class Momentum(Optim):
    def __init__(self,learning_rate=.001):
        super().__init__()
        self.learning_rate=learning_rate
        self.mu=0.99
        self.cache = dict()  
    
    def update(self,gates):
        for g in gates:
            if g.param_size==0:
                continue                
            if not (g in self.cache):
                self.cache[g]=(np.zeros(g.weight.shape),np.zeros(g.bias.shape))
                                
            v_weight,v_bias= self.cache[g]

            v_weight= self.mu * v_weight - self.learning_rate * g.dweight            
            g.weight += v_weight
            v_bias= self.mu * v_bias - self.learning_rate * g.dbias            
            g.bias += v_bias   
            
            self.cache[g]=(v_weight,v_bias)

                    