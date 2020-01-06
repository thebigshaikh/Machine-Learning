import numpy as np


def import_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    x = data[:, :2]
    y = data[:, 2]
    m = len(y)
    return data, x, y.reshape((len(y), 1)), m


d, X, Y, M = import_data("ex1data2.txt")
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

j_history = []

def normal_equation(X,Y):
    XT=np.transpose(X)
    x1=np.linalg.inv(XT.dot(X))
    x2=X.transpose().dot(Y)
    theta=x1.dot(x2)
    return theta


print(normal_equation(X,Y))
