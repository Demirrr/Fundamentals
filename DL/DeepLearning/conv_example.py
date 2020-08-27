import numpy as np
import matplotlib.pyplot as plt
from conv_util import zero_pad, select, show_slice
import torch
from torch import nn


plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

# create images
np.random.seed(1)
N = 1
N_H = 5
N_W = 5
n_C = 1
X = (np.random.randn(N, N_H, N_W, n_C) * 255).astype(np.uint8)
plt.imshow(X[0, :, :, :], interpolation='none')
plt.show()

X_padded = zero_pad(X, 1)
show_slice(select(X_padded, 0))
show_slice(select(X_padded, 0, rec_shape=(0, 4, 0, 4)))

# Convolution from scratch
x = select(X_padded, 0)[:, :, 0]
x.astype(float)

stride = 2
k = 3
conv = nn.Conv1d(in_channels=1,
                 kernel_size=(k, k),
                 out_channels=1, stride=stride)
x = torch.from_numpy(x)
x = x.float()
x = x.view(1, 1, 7, 7)

print(conv(x).detach().numpy())

w0 = conv.weight.data.numpy()
b0 = conv.bias.data.numpy()
x = x.numpy()

x = x.reshape(7, 7)
w0 = w0.reshape(3, 3)

def convolve_slice(X,W,b,stride=2):
    xH,xW=X.shape
    wH,wW=W.shape
    # https://cs231n.github.io/convolutional-networks/
    vH, vW = (xH - wH) // stride + 1, (xW - wW) // stride + 1

    assert vH==vW # convenience for the moment
    assert wH==wW
    k=wH
    v=np.zeros((vH,vW)) # output shape
    for i in range(vH):
        for j in range(vH):
            v[i,j]=np.sum(X[stride*i:(stride*i)+k,stride*j:(stride*j)+k] *W) +b
    return v


print(convolve_slice(x,w0,b0,2))
print('####')

xH,xW=7,7
stride=2
x=np.random.randn(xH,xW)
k = 3
conv = nn.Conv1d(in_channels=1,
                 kernel_size=(k, k),
                 out_channels=1, stride=stride)
# from numpy to torch
x_torch = torch.from_numpy(x)
x_torch = x_torch.float()
x_torch = x_torch.view(1, 1, 7, 7)
print(conv(x_torch).detach().numpy())

w0 = conv.weight.data.numpy()
b0 = conv.bias.data.numpy()
w0 = w0.reshape(3, 3)

print(convolve_slice(x,w0,b0,2))

