import numpy as np
import matplotlib.pyplot as plt


def loaddata(file, delimiter):
    return np.loadtxt(file, delimiter=delimiter)


def plotdata(data, label_x, label_y, label_pos, label_neg, axes=None):

    neg = data[:, 2] == 0
    pos = data[:, 2] == 1

    if axes == None:
        axes = plt.gca()

    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=30, lw=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], marker='x', c='r', s=30, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    # axes.legend(frameon=True, fancybox=True)
    axes.legend(loc='best')
    plt.show()


data = loaddata('data1.txt', ',')

X = np.c_[np.ones([data.shape[0], 1]), data[:, :2]]
y = np.c_[data[:, 2]]

# plotdata(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))


def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))

    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))

    if np.isnan(J[0]):
        return (np.inf)
    return J[0]

