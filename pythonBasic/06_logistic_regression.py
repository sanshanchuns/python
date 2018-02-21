import torch
from torch.autograd import Variable
import torch.nn.functional as F

x_data = Variable(torch.FloatTensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.FloatTensor([[0.0], [0.0], [1.0], [1.0]]))


class Model(torch.nn.Modelu):
    def __init__(self):
        pass

    def forward(self, x):
        pass


model = Model()

