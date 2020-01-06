import matplotlib.pyplot as plt
import numpy as np

popfood = np.loadtxt('ex1data1.txt', delimiter=',')
X, y = popfood[:, 0], popfood[:, 1]
m = y.size


def plot(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.title('Population vs Profit')
    plt.xlabel('Population * 10,000')
    plt.ylabel('Profit * 10.000')
    plt.show()


plot(X, y)
X = np.stack([np.ones(m), X], axis=1)


def cost_function(X, y, theta): 
    print(theta)
    m = y.size
    j1 = 0
    j1 = 1 / (2 * m) * np.sum(((X.dot(theta)).reshape((len(X)),1) - y.reshape((len(y), 1))) ** 2)
    return j1


a = cost_function(X, y, np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % a)
print('Expected cost value (approximately) 32.07\n')

a = cost_function(X, y, np.array([-1, 2]))
print('With theta = [-1, 2]\nCost computed = %.2f' % a)
print('Expected cost value (approximately) 54.24')


def gradient_descent(X, y, theta, alpha, iters):
    m = y.shape[0]
    theta = theta.copy()
    j_history = []

    for i in range(iters):
        print(i)
        print(theta)
        temp0 = theta[0] - (alpha / m) * np.sum((X.dot(theta)) - y)
        temp1 = theta[1] - (alpha / m) * np.sum(((X.dot(theta)) - y).dot(X))
        theta0 = temp0
        theta1 = temp1
        theta = [theta0, theta1]

        j_history.append(cost_function(X, y, theta))
    return theta, j_history


thetax, j_history = gradient_descent(X, y, np.array([0, 0]), 0.01, 2000)
plt.plot(j_history)
plt.show()

