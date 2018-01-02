import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = torch.pow(x, 2) + 0.05*torch.normal(torch.zeros(x.size()))

# plt.scatter(x.numpy(), y.numpy(), s=10, c=x.numpy(), cmap='rainbow')
# plt.show()

# Variable -> Tensor -> numpy, 默认是False
x, y = Variable(x), Variable(y, requires_grad=False)

# x.data 就是 tensor

net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for step in range(200):

    output = net(x)
    loss = loss_func(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(loss.data.numpy())

        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy(), s=10, c=x.data.numpy(), cmap='rainbow')
        plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=1)
        plt.pause(0.1)

plt.show()