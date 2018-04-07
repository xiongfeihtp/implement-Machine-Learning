import numpy as np


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    # 向量化计算
    dataMatIn = np.mat(dataMatIn)
    labelMat = np.mat(classLabels)
    m, n = np.shape(dataMatIn)
    # learning rate
    alpha = 0.001
    maxCycles = 500
    weights = np.random.randn((n, 1)) * 0.05
    for i in range(maxCycles):
        #向量化编程
        h = sigmoid(dataMatIn * weights)
        error = (labelMat - h)
        weights += alpha * dataMatIn.transpose() * error

    return weights
