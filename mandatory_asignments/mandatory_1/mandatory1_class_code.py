import pandas as pd
import math as m
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


# initialization
# dataset = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
weights = [-0.05, 0.02, -0.02]
# print(type(weights[0]))

# petal_length = dataset['petal length'].tolist()
# petal_width = dataset['petal width'].tolist()
# dataset2 = pd.DataFrame(dataset['petal length'].tolist(), dataset['petal width'].tolist())
# dataset3 = [[dataset.iloc[i, 0], dataset.iloc[i, 1]] for i in range(len(dataset))]
# weights = [random.uniform(-1,1) for i in range(len(dataset3))]

# training
print("weights before: ", weights)
weights = training(dataset, weights, 0.2, 5)
# print("weights after", weights)

# recall
# for sample in dataset3:
#     a = sigma(sample, weights)
#     print("Target=%d, Output=%d" % (sample[-1], a))

# PLOT

# print(dataset)
# dataset.plot(x='petal length', y='petal width', kind='scatter')
# plt.show()
