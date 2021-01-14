import numpy as np
from collections.abc import Iterable
from optim import ADAM
class Net:
    def __init__(self,optimizer=ADAM(),verbose=1):
        self.gates = []
        self.optimizer=optimizer
        self.losses=[]
        self.acc=[]
        
        self.verbose=verbose
        self.total_param=0
        
        self.eps=.001
    
    def describe(self):
        for th, g in enumerate(self.gates):
            print(f'[{th+1}. layer] => {g}') 
        print(f'Total param.:{self.total_param}')
        print(self.optimizer)


    def add(self, gate):
        self.total_param+=gate.param_size
        self.gates.append(gate)
        
    def add_from_iter(self, l):
        for gate in l:
            self.add(gate)

    def forward(self, inputs):
        if len(inputs)==1:
            inputs=np.expand_dims(inputs, axis=0)
        for g in self.gates:
            inputs = g.forward(inputs)
        return inputs

    def backward(self, dL):
        for g in reversed(self.gates):
            dL = g.backward(dL)
    
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle_per_epoch=True):
        assert inputs.shape[0] == targets.shape[0]
        if shuffle_per_epoch:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle_per_epoch:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]
    
    def update_gate(self,g):
        self.optimizer.update([g]) # workaround 

    def update(self,gates):
        for g in gates:
            if isinstance(g, Iterable):
                self.update(g)
            else:
                self.update_gate(g)
                
        
    def train(self,X,y,epoch=100,print_out_per_epoch=1,batchsize=256,shuffle_per_epoch=True):
        if self.verbose>0:
            print('Training starts.')
        for i in range(1, epoch+1):
            loss,acc=0,0
            for X_minibatch, y_minibatch in self.iterate_minibatches(X,y,batchsize,shuffle_per_epoch):

                Z=self.forward(X_minibatch)
                pred_prob_true_class=Z[range(len(Z)),y_minibatch]
                # batch accuracy
                acc  += (np.argmax(Z,axis=1)==y_minibatch).sum()
                
                loss += -np.log(pred_prob_true_class+self.eps).sum() #  

                # Compute Gradients of cross entropy loss w.r.t. predictions.
                dL_dZ = Z
                dL_dZ[range(len(Z)),y_minibatch] -= 1 
                dL_dZ/=len(dL_dZ) # important
                #dL_dZ=np.nan_to_num(dL_dZ,posinf=dL_dZ.max(),neginf=dL_dZ.min())

                self.backward(dL_dZ)
                self.update(self.gates)


            avg_acc=acc/len(X)
            avg_loss=loss/len(X)
            self.losses.append(avg_loss)
            self.acc.append(avg_acc)
            if i%print_out_per_epoch==0:
                    if self.verbose>0:
                        print(f'[Epoch:{i}]-[Avg.Loss:{avg_loss:.3f}]-[Avg.Acc:{avg_acc:.3f}]')