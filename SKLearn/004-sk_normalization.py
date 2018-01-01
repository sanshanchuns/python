from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification  #生产分类数据
from sklearn.svm import SVC # model Suport Vector Classifier
import matplotlib.pyplot as plt

# 列就是属性，就是维度，以下表示模型有三个属性
# a = np.array([[10, 2.7, 3.6],
#              [-100, 5, -2],
#              [120, 20, 40]], dtype=np.float32)
#
# print(a)
# print(preprocessing.scale(a))


x, y = make_classification(n_samples=300,
                           n_features=2,
                           n_redundant=0,
                           n_informative=2,
                           random_state=22,
                           n_clusters_per_class=1,
                           scale=100)

# plt.scatter(x[:, 0], x[:, 1], c=y)
# plt.show()

# x = preprocessing.minmax_scale(x, feature_range=(-1, 1)) # normalization
x = preprocessing.scale(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
clf = SVC()
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test)) #精确度
