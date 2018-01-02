import torch
import numpy as np

np_data = np.arange(6).reshape(2, 3)
print(np_data)
torch_data = torch.from_numpy(np_data)
print(torch_data)

torch2array = torch_data.numpy()
print(torch2array)

# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data) #32bit
# http://pytorch.org/docs/master/torch.html
print(tensor)
print(torch.abs(tensor))
print(np.sin(data))
print(torch.sin(tensor))

data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)

print(np.dot(data, data))
print(np.matmul(data, data))
print(torch.mm(tensor, tensor))

print(torch.__version__)