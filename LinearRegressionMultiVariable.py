import numpy as np
import matplotlib.pyplot as plt


def import_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    x = data[:, :2]
    y = data[:, 2]
    m = len(y)
    return data, x, y.reshape((len(y), 1)), m


def plot_data(xx1, xx2, yy):
    plt.scatter(xx1, xx2, yy)
    plt.show()


d, X, Y, M = import_data("ex1data2.txt")
plot_data(X[:, 0], X[:, 1], Y.ravel())
number_of_theta = d.shape[1]
X = np.concatenate([np.ones((M, 1)), X], axis=1)  # Adds a colimns of ones (X0) to X
theta_init = np.array([0 for i in range(number_of_theta)]).reshape(number_of_theta, 1)


def cost_function(sizem, xdata, ydata, theta):
    j = 0
    j = 1 / (2 * sizem) * np.sum(
        (xdata.dot(theta)
         - ydata) ** 2
    )
    return j


cost_value = cost_function(M, X, Y, theta_init)
print(cost_value)
j_history = []


def gradient_descent(alpha, size, X, Y, numofiters, theta_init):
    for i in range(numofiters):
        theta_init = theta_init - (alpha / size) * (np.dot(X, theta_init) - Y).dot(X)
        j_history.append(cost_function(size, X, Y, theta_init))
    return theta_init, j_history


th, jh = gradient_descent(0.01, M, X, Y, 2000, theta_init)
print(th, jh)
