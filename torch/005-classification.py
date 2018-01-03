import torch
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

data = np.ones((100, 2))
# a = np.random.randn(100, 2)
a = np.random.normal(1, 1, data.shape)
b = np.random.normal(-1, 1, data.shape)

x = np.vstack((a, b))
y = np.hstack((np.ones(100), np.zeros(100)))

# print(x.shape) #(200, 2)
# print(y.shape) #(200)

x, y = Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y))

# print(x.data.shape) # torch.Size([200, 2])

net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for step in range(300):

    output = net(x)
    loss = loss_func(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(loss)
