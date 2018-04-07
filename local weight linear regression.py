import numpy as np


def single_lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    row = xMat.shape[0]
    weights = np.mat(np.eye((row)))
    for i in range(row):
        diffMat = testPoint - xMat[i, :]
        weights[i, i] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))

    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular")
        return
    theta = xTx.I * (xMat.T* (weights * yMat))
    return testPoint * theta

def all_lwlrTest(testArr, xArr, yArr, k=1.0):
    y_hat = np.zeros(testArr.shape[0])
    for i, point in enumerate(testArr):
        y_hat[i] = single_lwlr(point, xArr, yArr, k)
        return y_hat
