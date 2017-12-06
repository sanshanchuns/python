import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

n_data = torch.ones(100, 1)

x0 = torch.normal(10*n_data, 1)
x1 = torch.normal(5*n_data, 1)

y0 = torch.ones(100)
y1 = torch.zeros(100)

x = torch.cat((x0, x1), dim=0)
y = torch.cat((y0, y1), dim=0).type(torch.LongTensor)  #标准值

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], s=10,
#             c=y.data.numpy(), cmap='rainbow')
# plt.ion()
# plt.show()

net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)

optimizer = torch.optim.Adam(net.parameters(), lr=0.02, betas=(0.9, 0.999))
loss_func = torch.nn.CrossEntropyLoss()

for i in range(100):
    output = net(x)
    loss = loss_func(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print(loss.data[0])

    if i == 99:
        plt.cla()

        # 这里有一个技巧
        # output 是一个 200x2 的张量
        # softmax() 之后是一个 200x2 的概率张量
        # torch.max( , 1) 是对合并两列之后的 200x1 的张量
        # torch.max( , 1)[1] 是max 所在的 index 组成的 200x1 的张量
        # print(F.softmax(output).data.numpy()[:10,])
        prediction = torch.max(F.softmax(output), 1)[1]
        pred_y = prediction.data.numpy()
        print(pred_y)
        target_y = y.data.numpy()
        plt.scatter(np.arange(200), x.data.numpy(), c=pred_y.reshape(200, 1), s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.show()
