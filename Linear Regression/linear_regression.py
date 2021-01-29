import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# x_vals = np.random.randint(low=1, high=30, size=100)
# y_vals = np.random.randint(low=1, high=30, size=100)

# x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, train_size=0.8)

# x_train = x_train.reshape(-1, 1)
# x_test = x_test.reshape(-1, 1)
# print(x_train.shape, y_train.shape)

# print(y_test.shape, y_pred.shape)

# acc_score = accuracy_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)

diabetes_X, diabetes_Y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]

x_train, x_test, y_train, y_test = train_test_split(diabetes_X, diabetes_Y, train_size=0.8)

lreg = linear_model.LinearRegression()

lreg.fit(x_train, y_train)
y_pred = lreg.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

plt.scatter(diabetes_X, diabetes_Y)
plt.show()
