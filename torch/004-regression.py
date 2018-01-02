import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = torch.pow(x, 2) + 0.04*torch.normal(torch.zeros(x.size()))

# plt.scatter(x.numpy(), y.numpy())
# plt.show()

loader = Data.DataLoader(
    dataset=Data.TensorDataset(data_tensor=x, target_tensor=y),
    batch_size=50,
    shuffle=True,
    num_workers=2,
)

# x, y = Variable(x), Variable(y)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(1, 10, 1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = torch.nn.MSELoss()

for epoch in range(10):
    for step, (b_x, b_y) in enumerate(loader):
        b_x, b_y = Variable(b_x), Variable(b_y)
        output = net(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(loss.data.numpy())

