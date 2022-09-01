import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


def step_func(z):
    return 1.0 if (z > 0) else 0.0


def perceptron(X, y, lr, epochs):
    """

    :param X:  Inputs
    :param y:  labels/target
    :param lr: learning rate
    :param epochs: n iterations
    :return:
    """
    m, n = X.shape

    # Initializing parapeters(theta) to zeros.
    # +1 in n+1 for the bias term.
    theta = np.zeros((n + 1, 1))

    # Empty list to store how many examples were
    # misclassified at every iteration.
    n_miss_list = []

    # Training.
    for epoch in range(epochs):

        # variable to store #misclassified.
        n_miss = 0

        # looping for every example.
        for idx, x_i in enumerate(X):

            # Insering 1 for bias, X0 = 1.
            x_i = np.insert(x_i, 0, 1).reshape(-1, 1)

            # Calculating prediction/hypothesis.
            y_hat = step_func(np.dot(x_i.T, theta))

            # Updating if the example is misclassified.
            if (np.squeeze(y_hat) - y[idx]) != 0:
                theta += lr * ((y[idx] - y_hat) * x_i)

                # Incrementing by 1.
                n_miss += 1

        # Appending number of misclassified examples
        # at every iteration.
        n_miss_list.append(n_miss)

    return theta, n_miss_list


def plot_decision_boundary(X, theta):
    # X --> Inputs
    # theta --> parameters

    # The Line is y=mx+c
    # So, Equate mx+c = theta0.X0 + theta1.X1 + theta2.X2
    # Solving we find m and c
    x1 = [min(X[:, 0]), max(X[:, 0])]
    m = -theta[1] / theta[2]
    c = -theta[0] / theta[2]
    x2 = m * x1 + c

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "r^")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    # plt.title(’Perceptron Algorithm’)
    plt.plot(x1, x2, 'y-')
    plt.show()


def plot_simple():
    fig = plt.figure(figsize=(10, 8))
    # print(X[:, 0][y == 0])
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^')  # 'r^' is the symbol red triangle
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')  # 'bs' is the symbol blue square
    # plt.xlabel("feature 1")
    # plt.ylabel("feature 2")
    # plt.title('Random Classification Data with 2 classes')
    plt.show()


if __name__ == '__main__':
    X, y = datasets.make_blobs(n_samples=150, n_features=2,
                               centers=2, cluster_std=1.05,
                               random_state=2)
    # X,y type: ndarray
    # plot_simple()

    theta, miss_l = perceptron(X, y, 0.5, 100)
    plot_decision_boundary(X, theta)