import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt

# torchvision 就是数据集
# data 就是分批处理器

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001

mnist_train = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor,
    download=False,
)

# print(mnist_train.train_data.shape)  #[60000, 28, 28]
# print(mnist_train.train_labels.shape)  #[60000]
# print(mnist_train.train_data[0])

# plt.imshow(mnist_train.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % mnist_train.train_labels[0])
# plt.show()

loader = Data.DataLoader(
    dataset=Data.TensorDataset(data_tensor=mnist_train.train_data, target_tensor=mnist_train.train_labels),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

mnist_test = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.ToTensor,
)

t_x = mnist_test.test_data[:2000]    #[2000, 28, 28]
t_y = mnist_test.test_labels[:2000]  #[2000]

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),  #(28, 28, 16)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), #(14, 14, 16)
        )
        self.conv2 = torch.nn.Sequential(

        )

    def forward(self, *input):
        pass



# cnn = torch.nn.Sequential(
#     # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
#     torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
#     #input(1, 28, 28)  output(16, 28, 28)
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(kernel_size=2), #output(16, 14, 14)
#
#     torch.nn.Conv2d(16, 32, 5, 1, 2),
#     #input(16, 28, 28) output(32, 14, 14)
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(2), #output(32, 7, 7)
#
#     torch.nn.Linear(32*7*7, 10)
# )

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

print(cnn)

# for epoch in range(EPOCH):
#     for step, (b_x, b_y) in enumerate(loader):

        # print(step, b_x[0], b_y)
    #     b_x, b_y = Variable(torch.unsqueeze(b_x, dim=1)), Variable(b_y)
    #     output = cnn(b_x)
    #     loss = loss_func(output, b_y)
    #     optimizer.zero_grad()
    #     loss.bacward()
    #     optimizer.step()
    #
    # print(loss)