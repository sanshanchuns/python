import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

x = np.linspace(-1, 1, 10)[:, np.newaxis]
index = np.random.randint(0, x.shape[0], 3)

print(index)
print(x)
print(x[index])
print(x[index].shape)