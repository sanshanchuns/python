import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

# torchvision 就是数据集
# data 就是分批处理器

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001

mnist_train = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)

# print(mnist_train.train_data.shape)  #[60000, 28, 28]
# print(mnist_train.train_labels.shape)  #[60000]
# print(mnist_train.train_data[0])

# plt.imshow(mnist_train.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % mnist_train.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(
    # dataset=Data.TensorDataset(
    #     data_tensor=mnist_train.train_data,
    #     target_tensor=mnist_train.train_labels
    # ), #[50, 28, 28] -> ByteTensor
    dataset=mnist_train,  #[50, 1, 28, 28] -> FloatTensor
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

mnist_test = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.ToTensor(),
)

t_x = mnist_test.test_data[:2000]    #[2000, 28, 28]
t_x = torch.unsqueeze(t_x, dim=1).type(torch.FloatTensor)/255.0 #[2000, 1, 28, 28] 值的范围 [0-1]
t_x = Variable(t_x)
t_y = mnist_test.test_labels[:2000]  #[2000]

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),  #(16, 28, 28)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), #(16, 14, 14) #stride影响输出
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),  #(32, 14, 14)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)  #(32, 7, 7)
        )
        self.out = torch.nn.Linear(32* 7* 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)    #(batch, 32, 7, 7)
        x = x.view(x.size(0), -1)  #(batch, 32*7*7)   保留 batch, 后面的维度合并
        output = self.out(x)
        return output

cnn = CNN()

# cnn = torch.nn.Sequential(
#     torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
#     #input(1, 28, 28)  output(16, 28, 28)
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(kernel_size=2), #output(16, 14, 14)
#
#     torch.nn.Conv2d(16, 32, 5, 1, 2),
#     #input(16, 28, 28) output(32, 14, 14)
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(2), #output(batch, 32, 7, 7)
#
#     torch.nn.Linear(32*7*7, 10)
# )

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

# print(cnn)

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        # b_x, b_y = Variable(torch.unsqueeze(x, 1).type(torch.FloatTensor)), Variable(y)
        b_x, b_y = Variable(x), Variable(y)
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            predict = torch.max(cnn(t_x[:10]), 1)[1] #[10, 1] tensor
            p = predict.data.squeeze() #[10] tensor
            print(p.numpy())
            print(t_y[:10].numpy())
