import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from sklearn import linear_model

# ---Get data---
data_file = pathlib.Path(__file__).resolve().parent.parent / \
    'data/andrewng/ex1data1.txt'
data = pd.read_csv(data_file, header=None, names=['Population', 'Profit'])
# print(data.describe())
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))

# ---Prepare data---
X = data.iloc[:, 0:data.shape[1]-1]
y = data.iloc[:, data.shape[1]-1]
# print(X.head())

X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)

# ---Optimize---
model = linear_model.LinearRegression()
model.fit(X, y)

# ---Plot---
prediction = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(X, prediction, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
