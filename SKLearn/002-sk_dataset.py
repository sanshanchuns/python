from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_x = loaded_data.data
data_y = loaded_data.target

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

model = LinearRegression()
model.fit(x_train, y_train)

# print(model.predict(x_test[:4]))
# print(y_test[:4])

x, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
plt.scatter(x, y)
plt.show()