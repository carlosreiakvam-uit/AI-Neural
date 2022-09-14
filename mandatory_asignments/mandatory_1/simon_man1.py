import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('./../data/iris.csv', header=None,
                      names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

# Remove class not used for classification
dataset.drop(index=dataset.index[dataset['class'] == 'Iris-versicolor'], inplace=True)

# Remove columns not used for classification
dataset.drop('sepal length', axis='columns', inplace=True)
dataset.drop('sepal width', axis='columns', inplace=True)

# Assigning 0 and 1 to the remaining classes
dataset.loc[dataset['class'] == 'Iris-setosa', dataset.columns == 'class'] = 0
dataset.loc[dataset['class'] == 'Iris-virginica', dataset.columns == 'class'] = 1

# print(dataset.to_string())

numpy_data = dataset.to_numpy()
print(numpy_data.tostring())


# from lecture on boolean operator OR
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
        for x in data:
            activation = sigma(x, w)
            error = x[-1] - activation
            w[0] = w[0] + mu * error
            for i in range(len(x) - 1):
                w[i + 1] = w[i + 1] + mu * error * x[i]
    return w

# initialization
# dataset = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
weights = [-0.05, 0.02, -0.02]


# training
# print("weights before: ", weights)
# weights = training(numpy_data, weights, 0.2, 5)
# print("weights after: ", weights)

# recall
#for sample in numpy_data:
#    a = sigma(sample, weights)
#    print("Target=%d, Output=%d" % (sample[-1], a))