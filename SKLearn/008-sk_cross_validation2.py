from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
x = digits.data
y = digits.target

train_sizes, train_loss, test_loss = learning_curve(
    SVC(gamma=0.001), x, y, cv=10, scoring='neg_mean_squared_error',
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1]  #学习进度在10%, 25%... 记录一下所有的loss
)

train_loss_mean = -np.mean(train_loss, axis=1) #10个维度合一， 10列合一
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label='Training')
plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label='Cross-validation')

plt.xlabel('Training Examples')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()

# 红线的误差曲线一直都是很低的，因为是 training 是有overfit倾向的
# 绿线的误差曲线是慢慢降低的，因为在纠正overfit
