import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.normal(torch.ones(x.size()), 1)

loader = Data.DataLoader(
    dataset=Data.TensorDataset(data_tensor=x, target_tensor=y),
    batch_size=5,
    shuffle=True,
    num_workers=2,
)

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(1, 10)
#         self.predict = torch.nn.Linear(10, 1)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x
#
# net = Net()

def save():
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )

    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for i in range(100):
        output = net(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #训练完成，保存结果
    torch.save(net, 'net.pkl')
    torch.save(net.state_dict(), 'net_params.pkl')

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)

def restore_net():
    net2 = torch.load('net.pkl')
    output = net2(x)

    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)

def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )

    net3.load_state_dict(torch.load('net_params.pkl'))
    output = net3(x)

    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)
    plt.show()


save()
restore_net()
restore_params()