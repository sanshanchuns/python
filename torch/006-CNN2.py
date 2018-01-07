import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision  #dataset mnist
import torch.utils.data as Data  #mini batch
import numpy as np
import os

BATCH_SIZE = 50
LR = 0.001
EPOCH = 1

DOWNLOAD = False
if not os.path.exists('./mnist') or not os.listdir('./mnist'):
    DOWNLOAD = True

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

test_data = torchvision.datasets.MNIST('./mnist', False, torchvision.transforms.ToTensor())

test_x = test_data.test_data[:10]  #[10, 28, 28], ByteTensor
test_y = test_data.test_labels[:10] #[10]

test_x = Variable(torch.unsqueeze(test_x, 1).type(torch.FloatTensor))  #[10, 1, 28, 28], FloatTensor


#数据集处理完毕

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2), #[batch, 16, 28, 28]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), #[batch, 16, 14, 14]
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), #[batch, 32, 7, 7]
        )
        self.out = torch.nn.Linear(32*7*7, 10)

    def forward(self, input):
        input = self.conv1(input)
        input = self.conv2(input)  #[batch, 32, 7, 7]
        input = input.view(input.size(0), -1)
        output = self.out(input)
        return output

cnn = CNN()
# print(cnn)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):

        b_x, b_y = Variable(x), Variable(y) # [50, 1, 28, 28], [50]
        output = cnn(b_x) #[50, 10]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            predict = torch.max(cnn(test_x), 1)[1].data.squeeze() #[10, 1]tensor
            print(predict.numpy())
            print(test_y.numpy())