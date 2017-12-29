from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


iris = load_iris()
iris_x = iris.data
iris_y = iris.target

# iris_x = preprocessing.scale(iris_x) #正则normalization

# x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.3)

# knn = KNeighborsClassifier(n_neighbors=5) #考虑数据点附近5个点的平均值

# knn.fit(x_train, y_train)
# print(knn.score(x_test, y_test))

# scores = cross_val_score(knn, iris_x, iris_y, cv=5, scoring='accuracy')
# print(scores.mean()) # 5组

k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # scores = cross_val_score(knn, iris_x, iris_y, cv=10, scoring='accuracy') # for classification
    # k_scores.append(scores.mean())
    losses = -cross_val_score(knn, iris_x, iris_y, cv=10, scoring='neg_mean_squared_error') # for regression
    k_scores.append(losses.mean())

plt.plot(k_scores)
plt.show()




