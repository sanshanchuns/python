import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x_data = Variable(torch.FloatTensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.FloatTensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_predict = self.linear(x)
        return y_predict

model = Model()

loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    predict = model(x_data)
    l = loss(predict, y_data)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    print(epoch, l.data[0])


output = model(Variable(torch.FloatTensor([4.0])))
print('predict after traing ', output.data[0])
