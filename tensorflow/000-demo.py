import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(2, 2, 2)
# r, g, b = x[:,:,0], x[:,:,1], x[:,:,2]
# gray = 0.29*r + 0.58*g + 0.11*b
# gray = np.mean(x, -1)
# print(x)
# print(gray)
# print(x.flatten())
import tensorflow as tf

steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
# use sin predicts cos
x = np.sin(steps)    # shape (batch, time_step, input_size)
y = np.cos(steps)
z = np.tan(steps)

plt.plot(x)
plt.plot(y)
plt.plot(z)
plt.show()









