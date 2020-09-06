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


def convolve_step(x, w, b, S):
    """ convolve (slide) over all spatial location."""
    xD, xH, xW = x.shape
    wD, wH, wW = w.shape
    assert xH == xW and wH == wW and xD == wD
    assert (xW - wW) % S == 0
    num_conv_opt = int((xW - wW) / S + 1)
    Z = np.zeros((num_conv_opt, num_conv_opt))
    F = wH
    for i in range(num_conv_opt):  # vertical
        for j in range(num_conv_opt):  # horiz

            x_loc = x[:,  # depth.
                    S * i: (S * i) + F,
                    S * j: (S * j) + F]
            Z[i, j] = np.sum(x_loc * w) + b
    return Z


def convolve(X, W, b, S, pad=0):
    # parameter check
    xN, xD, xH, xW = X.shape
    wN, wD, wH, wW = W.shape
    assert wH == wW
    assert (xH - wH) % S == 0
    assert (xW - wW) % S == 0

    zH, zW = (xH - wH) // S + 1, (xW - wW) // S + 1

    zD, zN = wN, xN
    Z = np.zeros((zN, zD, zH, zW))

    for n in range(zN):
        x = X[n]
        for d in range(zD):
            # convolve d.th kernel on n.th input with S.
            Z[n, d, :, :] = convolve_step(x, W[d], b[d], S)

    return Z


xN, xD, xH, xW = 3, 3, 7, 7
X = np.random.randn(xN, xD, xH, xW)
# kernel init
nW, k, s = 2, 5, 2
conv = nn.Conv1d(in_channels=xD, out_channels=nW, kernel_size=(k, k), stride=s)

x_torch = torch.from_numpy(X)
x_torch = x_torch.float()
res = conv(x_torch)
print(res)
W = conv.weight.data.detach().numpy()
b = conv.bias.data.detach().numpy()
Z = convolve(X, W, b, s)
print(Z)


def p(x, F, S):
    """ convolve (slide) over all spatial location."""
    xD, xH, xW = x.shape
    assert (xW - F) % S == 0
    num_opt = int((xW - F) / S + 1)
    Z = np.zeros((xD, num_opt, num_opt))
    for i in range(num_opt):  # vertical
        for j in range(num_opt):  # horiz
            for d in range(xD):
                Z[d, i, j] = x[d,  # depth
                             S * i: (S * i) + F,
                             S * j: (S * j) + F].max()
    return Z


def pooling(X, k, S):
    # parameter check
    xN, xD, xH, xW = X.shape
    assert (xH - k) % S == 0
    assert (xW - k) % S == 0

    zH, zW = (xH - k) // S + 1, (xW - k) // S + 1
    zN, zD = xN, xD
    Z = np.zeros((zN, zD, zH, zW))

    for n in range(zN):
        x = X[n]
        Z[n, :, :, :] = p(x, k, S)

    return Z


v = res.data.detach()
m = nn.MaxPool2d(2, stride=s)

print(m(v))

pooling(v.numpy(), 2, s)


def convolve_forward_step(x, W, b, S):
    """
    x.shape = xD, xH, xW
    W.shape = wN,wD, wH, wW

    convolve (slide) over all spatial location."""
    xD, xH, xW = x.shape
    wN, wD, wH, wW = W.shape
    assert (xW - wW) % S == 0
    num_conv_opt = int((xW - wW) / S + 1)
    Z = np.zeros((wN, num_conv_opt, num_conv_opt))
    F = wH

    for d in range(wN):  # .th kernel
        for i in range(num_conv_opt):  # vertical
            for j in range(num_conv_opt):  # horiz

                x_loc = x[:,  # depth.
                        S * i: (S * i) + F,
                        S * j: (S * j) + F]
                Z[d, i, j] = np.sum(x_loc * W[d]) + b[d]
    return Z


def convolve_forward(X, W, b, S, pad=0):
    # parameter check
    xN, xD, xH, xW = X.shape
    wN, wD, wH, wW = W.shape
    assert wH == wW
    assert (xH - wH) % S == 0
    assert (xW - wW) % S == 0

    zH, zW = (xH - wH) // S + 1, (xW - wW) // S + 1

    zD, zN = wN, xN
    Z = np.zeros((zN, zD, zH, zW))

    for n in range(zN):
        Z[n, :, :, :] = convolve_forward_step(X[n], W, b, S)

    cache = (X, W, b, S)
    return Z, cache


xN, xD, xH, xW = 2, 3, 7, 7
X = np.random.randn(xN, xD, xH, xW)
# kernel init
nW, k, s = 2, 1, 2
conv = nn.Conv1d(in_channels=xD, out_channels=nW, kernel_size=(k, k), stride=s)

x_torch = torch.from_numpy(X)
x_torch = x_torch.float()
res = conv(x_torch)
print(res)

W = conv.weight.data.detach().numpy()
b = conv.bias.data.detach().numpy()

Z, _ = convolve_forward(X, W, b, s)
print(Z.shape)
print(Z)