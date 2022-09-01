import pandas as pd
import matplotlib.pyplot as plt

# read data
dataset = pd.read_csv('data/iris.csv', header=None,
                      names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
# drop iris-versicolor
dataset.drop(index=dataset.index[dataset['class'] == 'iris-versicolor'])

# drop columns that should not be compared
dataset.drop('sepal length', axis='columns', inplace=True)
dataset.drop('sepal width', axis='columns', inplace=True)

# Assign 0 and one to remaining datasets
# loc arguments: rows, columns
dataset.loc[dataset['class'] == 'iris-setosa', dataset.columns == 'class'] = 0
dataset.loc[dataset['class'] == 'iris-virginica', dataset.columns == 'class'] = 1
# dataset.loc[[0, 1, 2], :] = 0

# PLOT
# dataset.plot(x='petal length', y='petal width', kind='scatter')
# binary_dataset.plot(x='petal length', y='petal width', kind='scatter')
# plt.show()

# TESTING PRINTS
# print(binary_dataset)
# print(dataset.loc[[0, 1, 2], :])
# print(dataset)
print(dataset)
