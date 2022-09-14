import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data
dataset = pd.read_csv('./../data/iris.csv', header=None,
                      names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

# Remove class not used for classification
dataset.drop(index=dataset.index[dataset['class'] == 'Iris-versicolor'], inplace=True)

# drop columns that should not be compared
dataset.drop('sepal length', axis='columns', inplace=True)
dataset.drop('sepal width', axis='columns', inplace=True)

# Assign 0 and 1 to remaining datasets, representing the correct outputs
dataset.loc[dataset['class'] == 'Iris-setosa', dataset.columns == 'class'] = 0
dataset.loc[dataset['class'] == 'Iris-virginica', dataset.columns == 'class'] = 1

print(dataset.to_string())


# neuron activation
def sigma(x, w):
    for i in range(len(x) - 1):  # excluding last element which is output
        w[0] += w[i + 1] * x[i]  # summing activations
    return 1.0 if w[0] >= 0.0 else 0.0


# training function
def training(data, w0, mu, T):
    w = w0
    for t in range(T):
        for x in data:  # row
            y = x[-1]
            activation = sigma(x, w)
            # error = correct_output - activation
            error = activation - y
            # if error != 0:
            #     print("registered error")
            # w[0] = w[0] + mu * error
            for i in range(len(x) - 1):  # updating weights excluding output column
                w[i] = w[i] + mu * error * x[i]  # error is either 0 or 1, only updates on activation = 1
                # w[i + 1] = w[i + 1] + mu * error * x[i]  # error is either 0 or 1, only updates on activation = 1
    return w


# initialization
w = [-0.1, 0.1, 0.1]
numpy_data = dataset.to_numpy()

# training
print("weights before: ", w)
w = training(numpy_data, w, 0.2, 5)
print("weights after", w)

# recall
for sample in numpy_data:
    a = sigma(sample, w)
    print("Target=%d, Output=%d" % (sample[-1], a))

lin_x = np.linspace(0, 1, 100)
lin_x2 = np.linspace(1, 7, 100)
w1, w2, b = w[0], w[1], -2.5

x_intercept = (0, -b / w2)
y_intercept = (-b / w1, 0)
m = -(b / w2) / (b / w1)
y = -(-b / w2) / (b / w1) * lin_x2 + (-b / w2)

# PLOT
dataset.plot(x='petal length', y='petal width', kind='scatter')
plt.plot(lin_x2, y)
plt.show()
