import numpy as np
import torch

outs = []

for i in range(10):
    outs.append(torch.ones(1))

print(outs)
print(torch.stack(outs))
print(torch.stack(outs, dim=1))

