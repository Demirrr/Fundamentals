import torch
import torch.nn as nn
from res.plot_lib import set_default, show_scatterplot, plot_bases
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, title, axis

# Set style (needs to be in a new cell)
set_default()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# generate some points in 2-D space
n_points = 1000
X = torch.randn(n_points, 2).to(device)  # N(0,1)
colors = X[:, 0]
show_scatterplot(X, colors, title='X')
OI = torch.cat((torch.zeros(2, 2), torch.eye(2))).to(device)
plot_bases(OI)
plt.show()


model = nn.Sequential(
        nn.Linear(2, 2, bias=False),
        nn.Tanh()
).to(device)

for s in range(1, 6):
    W = s * torch.eye(2)
    model[0].weight.data.copy_(W)
    Y = model(X).data
    show_scatterplot(Y, colors, title=f'f(x), s={s}')
    plot_bases(OI, width=0.01)
    plt.show()