import torch
import torchvision
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

EPOCH = 10
LR = 0.005
BATCH_SIZE = 64
N_TEST_IMG = 5
ACT = torch.nn.Tanh()

torch.manual_seed(1)    # reproducible

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

class AutoEncode(torch.nn.Module):
    def __init__(self):
        super(AutoEncode, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28*28, 128),
            ACT,
            torch.nn.Linear(128, 64),
            ACT,
            torch.nn.Linear(64, 12),
            ACT,
            torch.nn.Linear(12, 3),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 12),
            ACT,
            torch.nn.Linear(12, 64),
            ACT,
            torch.nn.Linear(64, 128),
            ACT,
            torch.nn.Linear(128, 28*28),
            torch.nn.Sigmoid(),  #[0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

auto_encoder = AutoEncode()
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)

fig, ax = plt.subplots(2, 5, figsize=(5, 2))

view_data = Variable(train_data.train_data[:5]).view(-1, 28*28).type(torch.FloatTensor)

for i in range(5):
    # print(train_data.train_data[i]) # ByteTensor [28, 28]
    # print(train_data.train_labels) # LongTensor [60000]
    ax[0][i].imshow(view_data[i].data.numpy().reshape(28, 28), cmap='gray')
    ax[0][i].set_xticks(())
    ax[0][i].set_yticks(())


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        # print(x) #[50, 1, 28, 28] FloatTensor
        # print(y) #[50] LongTensor

        x = Variable(x)
        encoded, decoded = auto_encoder(x.view(-1, 28*28))
        # print(decoded) # [50, 28*28]
        loss = loss_func(decoded, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:

            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])

            en, de = auto_encoder(view_data)

            for i in range(5):
                ax[1][i].clear()
                ax[1][i].imshow(de[i].data.numpy().reshape(28, 28), cmap='gray')
                ax[1][i].set_xticks(())
                ax[1][i].set_yticks(())
            plt.pause(.1)

plt.show()



