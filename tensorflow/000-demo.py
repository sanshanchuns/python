import numpy as np

x = np.random.rand(2, 2, 2)
# r, g, b = x[:,:,0], x[:,:,1], x[:,:,2]
# gray = 0.29*r + 0.58*g + 0.11*b
# gray = np.mean(x, -1)
# print(x)
# print(gray)
# print(x.flatten())
import tensorflow as tf

step = 1
start, end = step * np.pi, (step+1)*np.pi   # time range
# use sin predicts cos
steps = np.linspace(start, end, 10)
x = np.sin(steps)[np.newaxis, :, np.newaxis]    # shape (batch, time_step, input_size)
y = np.cos(steps)[np.newaxis, :, np.newaxis]

print(np.sin(steps))
print(np.cos(steps))
print(x)
print(y)









