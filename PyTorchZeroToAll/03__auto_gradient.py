import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [1, 4, 9]

w = Variable(torch.Tensor([1.0]), requires_grad=True)
w_his = []
loss_his = []

def forward(x):
    return w*x

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)*(y_pred - y)

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()

        w.data = w.data - 0.01*w.grad.data

        w.grad.data.zero_()

print(w.data)

plt.plot(w_his, loss_his)
plt.show()


