# import torch
# from torch.autograd import Variable
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import torch.utils.data as Data
#
# #fake data
# x = torch.unsqueeze(torch.linspace(-1, 1, 300), dim=1)  #(300, 1)
# y = x.pow(2) + 0.2*torch.normal(torch.zeros(x.size()))
#
# x, y = Variable(x), Variable(y)
#
# #batch_data_set
# # loader = Data.DataLoader(
# #     dataset=Data.TensorDataset(data_tensor=x, target_tensor=y),
# #     shuffle=True,
# #     batch_size=32,
# #     num_workers=2,
# # )
#
# # plt.scatter(x, y)
# # plt.show()
#
# #create layer
# net = torch.nn.Sequential(
#     torch.nn.Linear(1, 10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10, 1),
# )
#
# #calculate loss
# loss_func = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
# losses_his = []
#
# # for epoch in range(12):
# #     for step, (batch_x, batch_y) in enumerate(loader):
# #         xs, ys = Variable(batch_x), Variable(batch_y)
# #         print('step %d' % step)
# #         output = net(xs)
# #         loss = loss_func(output, ys)
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()
# #
# #         print('loss %f' % loss)
#
# for i in range(100):
#     output = net(x)
#     loss = loss_func(output, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     print(loss.data.numpy())
#     losses_his.append(loss.data.numpy())
#
#     if i % 2 == 0:
#         plt.figure(1, figsize=(10, 3))
#         plt.clf()
#         plt.subplot(121)
#         plt.scatter(x.data.numpy(),
#                     y.data.numpy(),
#                     s=10,
#                     c=x.data.numpy(),
#                     cmap='rainbow',
#                     )
#         plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=2)
#
#         plt.subplot(122)
#         plt.plot(losses_his, label='SGD')
#         plt.legend(loc='best')
#         plt.xlabel('Steps')
#         plt.ylabel('Loss')
#         plt.ylim((0, 0.2))
#         plt.pause(0.1)
#
# plt.pause(3)

import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.05*torch.normal(torch.zeros(x.size()))

loader = Data.DataLoader(
    dataset=Data.TensorDataset(data_tensor=x, target_tensor=y),
    batch_size=50,
    shuffle=True,
    num_workers=2,
)

# plt.scatter(x.data.numpy(), y.data.numpy(), c=x.data.numpy(), s=10, cmap='rainbow')
# plt.show()

net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

for epoch in range(10):
    for step, (b_x, b_y) in enumerate(loader):
        b_x, b_y = Variable(b_x), Variable(b_y)

        output = net(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(epoch, loss.data.numpy())