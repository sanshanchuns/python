import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [1.0, 4.0, 9.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)

def forward(x):
    return w*x

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)*(y_pred - y)

for epoch in range(10):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        w.data = w.data - 0.01*w.grad.data
        w.grad.data.zero_()

    print("process: ", epoch, l.data[0])

