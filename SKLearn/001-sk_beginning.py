import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

#测试比例 30%(x_test, y_test)
#顺序是 train, test --> train.x test.x train.y test.y
x_train, x_test, y_train, y_test = \
    train_test_split(iris_x, iris_y, test_size=0.3)

# print(y_train) # shuffle 过的

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print(knn.predict(x_test))
print(y_test)


