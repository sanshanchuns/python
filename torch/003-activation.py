import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-5, 5, 200), dim=1)
x = Variable(x)

x_np = x.data.numpy()

print(x_np.shape)

y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

plt.figure(1, figsize=(12, 12))

plt.subplot(221)
plt.plot(x_np, y_relu, 'r-', lw=1, label='relu')
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, 'r-', lw=1, label='sigmoid')
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, 'r-', lw=1, label='tanh')
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, 'r-', lw=1, label='softplus')
plt.legend(loc='best')

plt.show()