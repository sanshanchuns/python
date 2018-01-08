import torch
from torch.autograd import Variable
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
import os

EPOCH = 1
TIME_STEP = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
LR = 0.01

train_data = torchvision.datasets.MNIST('./mnist', True, torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = torchvision.datasets.MNIST('./mnist', False, torchvision.transforms.ToTensor())
test_x = Variable(torch.unsqueeze(test_data.test_data, 1)).type(torch.FloatTensor)[:10]/255.0
test_y = test_data.test_labels.numpy().squeeze()[:10]

print()
