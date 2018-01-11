import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 10      # rnn time step / image height
INPUT_SIZE = 1      # rnn input size / image width
LR = 0.02           # learning rate
DOWNLOAD_MNIST = False  # set to True if haven't download the data


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        # (batch, time_step, input_size) (batch, 10, 1)
        # (batch, time_step, hidden_size) (batch, 10, 32)

        self.out = torch.nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state) #(batch, 10, 32)
        outs = []

        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :])) #input(batch, 1, 32) output(batch, 1)
        output = torch.stack(outs, dim=1) #在dim=1的维度上做concatenate  output(batch, 1, 10)
        return output, h_state

rnn = RNN()
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)

h_state = None
for i in range(50):
    start, end = np.pi*i, np.pi*(i+1)
    steps = np.linspace(start, end, 10, dtype=np.float32)

    # input (batch, 10, 1)
    x = Variable(torch.from_numpy(np.sin(steps)[np.newaxis, :, np.newaxis]))
    # input (batch, 1, 10)
    y = Variable(torch.from_numpy(np.cos(steps)))

    # output (batch, 1, 10)
    predict, h_state = rnn(x, h_state)
    h_state = Variable(h_state.data)

    loss = loss_func(predict, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, predict.view(-1).data.numpy(), 'r-')
    plt.plot(steps, y.view(-1).data.numpy(), 'b-')
    plt.pause(.1)







