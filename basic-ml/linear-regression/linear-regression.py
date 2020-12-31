import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

# ---Get data---
data_file = pathlib.Path(__file__).resolve().parent.parent / \
    'data/andrewng/ex1data1.txt'
data = pd.read_csv(data_file, header=None, names=['Population', 'Profit'])
# print(data.describe())
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))

# ---Prepare data---
data.insert(0, 'Ones', 1)
X = data.iloc[:, 0:data.shape[1]-1]
y = data.iloc[:, data.shape[1]-1]
# print(X.head())

X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)
theta = np.zeros([X.shape[1], 1])


# ---Optimization functions---
def calc_cost(X, y, theta):
    return np.sum(np.power((X @ theta) - y, 2)) / (2 * X.shape[0])


def calc_gradient(X, y, theta):
    grad = ((X @ theta) - y) * X
    return (np.sum(grad, axis=0) / X.shape[0]).reshape(-1, 1)


def gradient_descent(X, y, theta, alpha=0.01, iterations=1000):
    for i in range(iterations):
        theta = theta - alpha * calc_gradient(X, y, theta)
    return theta


# ---Optimize---
initial_cost = calc_cost(X, y, theta)
final_theta = gradient_descent(X, y, theta)
final_cost = calc_cost(X, y, final_theta)
print("Initial cost: {}".format(initial_cost))
print("Final cost: {}".format(final_cost))
print("Final theta: \n{}".format(final_theta))

# ---Plot---
x = np.linspace(data.Population.min(), data.Population.max(), 100)
prediction = final_theta[0] + (final_theta[1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, prediction, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
