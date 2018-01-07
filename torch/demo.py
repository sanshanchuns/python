import numpy as np
import torch

a = torch.ones(2, 3, 1, 2)

b = a.view(a.size(0), -1)

print(a.size(1))