import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# neuron activation
def sigma(x, w):
    activation = 0
    for i in range(len(x)):  # excluding last element which is output
        activation += w[i] * x[i]  # summing activations
    return 1.0 if activation >= 0.0 else 0.0


# training function
def training(data, w, mu, T):
    weight_history = []
    for t in range(T):
        for x in data:  # row
            y, x = x[-1], x[:-1]
            activation = sigma(x, w)
            error = activation - y
            for i in range(len(x)):  # updating weights excluding output column
                w[i] = w[i] + mu * error * x[i]  # error is either 0 or 1, only updates on activation = 1
            weight_history.append(w)
    # print(pd.DataFrame(weight_history).to_string())
    return w


def plot_decision_boundary_line(w,numpy_data):
    for i in np.linspace(np.amin(numpy_data[:, :1]), np.amax(numpy_data[:, :1]), 100):
        slope = -(w[0] / w[2]) / (w[0] / w[1])
        intercept = -w[0] / w[2]

        # y = ax+b, (a=slope and b=intercept)
        y = (slope * i) + intercept
        plt.plot(i, y, markersize=2, marker=".")

if __name__ == '__main__':

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

    # initialization
    w = [0.1, 0.1]
    numpy_data = dataset.to_numpy()

    # training
    print("weights before: ", w)
    w = training(numpy_data, w, 0.2, 10)
    print("weights after", w)

    # recall
    # for sample in numpy_data:
    #     a = sigma(sample[:-1], w)  # determining errors using new weights
    #     print("Target = %d, prediction = %d" % (sample[-1], a))

    # plot_decision_boundary_line(w,numpy_data)



    # Scatter plot of data
    # dataset.plot(x='petal length', y='petal width', kind='scatter')

    # Show plot
    plt.show()

    # old junk
    # lin_x = np.linspace(0, 1, 100)
    # lin_x2 = np.linspace(1, 7, 100)
    # w1, w2, b = w[0], w[1], -2.5
    #
    # x_intercept = (0, -b / w2)
    # y_intercept = (-b / w1, 0)
    # m = -(b / w2) / (b / w1)
    # y = -(-b / w2) / (b / w1) * lin_x2 + (-b / w2)
