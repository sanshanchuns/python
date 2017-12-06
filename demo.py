#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

for i in range(100):
    z = i*10

print(z)

# x = np.arange(200)
#
# print(x)
#
# plt.scatter(x, x, s=10, c=np.arange(200), cmap='rainbow')
# plt.show()

# n_data = torch.ones(100, 2)
# x = torch.normal(-2*n_data, 1)
# y = torch.normal(2*n_data, 1)

# plt.plot(x.numpy(), label=y[0])
# plt.scatter(x.numpy(), y.numpy(), c=x.numpy(), cmap='rainbow', s=10)
# plt.legend(loc='best')
# plt.show()














