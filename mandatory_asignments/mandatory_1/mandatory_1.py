import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# read data
dataset = pd.read_csv('./../data/iris.csv', header=None,
                      names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

# drop iris-versicolor
dataset.drop(index=dataset.index[dataset['class'] == 'Iris-versicolor'], inplace=True)

# drop columns that should not be compared
dataset.drop('sepal length', axis='columns', inplace=True)
dataset.drop('sepal width', axis='columns', inplace=True)

# Assign 0 and one to remaining datasets
dataset.loc[dataset['class'] == 'Iris-setosa', dataset.columns == 'class'] = 0
dataset.loc[dataset['class'] == 'Iris-virginica', dataset.columns == 'class'] = 1


# neuron activation
def sigma(x, w):
    activation = w[0]
    for i in range(len(x) - 1):
        activation += w[i + 1] * x[i]
    return 1.0 if activation >= 0.0 else 0.0


# training function
def training(data, w0, mu, T):
    w = w0
    for t in range(T):
        for x in data:  # row
            activation = sigma(x, w)
            error = x[-1] - activation
            w[0] = w[0] + mu * error
            for i in range(len(x) - 1):  # updating weights
                w[i + 1] = w[i + 1] + mu * error * x[i]
    return w


def plot_decision_boundary_line(w, data):
    min, max = np.amin(data[:, :1]), np.amax(data[:, :1])
    x = np.linspace(min, max, 100)
    slope = -(w[0] / w[2]) / (w[0] / w[1])
    intercept = -w[0] / w[2]
    y = (slope * x) + intercept
    plt.plot(x, y)


# initialization
np_dataset = dataset.to_numpy()
# weights = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] # used for initial testing
weights = [-0.7, -0.2, 0.5]  # These weights seem to fit the dataset well

# training
weights = training(np_dataset, weights, 0.2, 5)

# recall
passed = True
for sample in np_dataset:
    a = sigma(sample, weights)
    if a != sample[-1]: passed = False
print("Perceptron classification passed:", passed)

# PLOT
dataset.plot(x='petal length', y='petal width', kind='scatter')
plot_decision_boundary_line(weights, np_dataset)
plt.show()
