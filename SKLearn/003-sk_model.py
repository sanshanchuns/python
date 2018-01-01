from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

loaded_data = datasets.load_boston()
data_x = loaded_data.data
data_y = loaded_data.target

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

model = LinearRegression()
model.fit(x_train, y_train)

# print(model.predict(x_test[:4]))
# print(y_test[:4])

# y = 0.1x + 0.3, 即线性回归的参数
print(model.coef_) # 0.1
print(model.intercept_) # 0.3

print(model.get_params()) # 模型参数

print(model.score(data_x, data_y)) #评判model的学习成绩 R^2 coefficient determination
