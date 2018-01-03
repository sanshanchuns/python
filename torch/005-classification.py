import torch
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

# 尝试 numpy 转换 tensor 再 Variable

# data = np.ones((100, 2))
# a = np.random.randn(100, 2)
# a = np.random.normal(1, 1, data.shape)
# b = np.random.normal(-1, 1, data.shape)
#
# x = np.vstack((a, b))
# y = np.hstack((np.ones(100), np.zeros(100)))

# print(x.shape) #(200, 2)
# print(y.shape) #(200)

data = torch.ones(100, 2)  #这里的2一方面是画图，另一方面是神经网络的输入
a = torch.normal(2*data, 1)
b = torch.normal(-2*data, 1)
c = torch.normal(6*data, 1)
d = torch.normal(10*data, 1)

x = torch.cat((a, b, c, d), dim=0) #这里的0表示行的追加
y = torch.cat((torch.ones(100), torch.zeros(100), 2*torch.ones(100), 3*torch.ones(100))).type(torch.LongTensor)
# 这里的标签是任意顺序的, 现在是 (1, 0, 2, 3), 同时也是神经网络的输出

# print(x.shape) #(200, 2)
# print(y.shape) #(200)

x, y = Variable(x), Variable(y)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 4),
)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for step in range(300):

    output = net(x)
    loss = loss_func(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        # print(loss.data.numpy())
        pred = torch.max(output, 1)[1] # 第一个1是合并列，第二个1是index所在的position
        accuracy = sum(pred.data.numpy() == y.data.numpy()) / 400

        plt.cla()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred.data.numpy(), cmap='rainbow')
        plt.text(8, -4, 'Accuracy:%.2f' % accuracy, fontdict={'color':'red', 'size':15})
        plt.pause(.3)

plt.show()