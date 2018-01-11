import torch
import torchvision
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt

EPOCH = 10
LR = 0.001
BATCH_SIZE = 50

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
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 12),
            torch.nn.Tanh(),
            torch.nn.Linear(12, 3),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 12),
            torch.nn.Tanh(),
            torch.nn.Linear(12, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 128),
            torch.nn.Tanh(),
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


