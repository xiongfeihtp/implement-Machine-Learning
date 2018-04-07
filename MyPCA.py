import numpy as np


def pca(dataMat, topNfeat):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    # rowvar = True 表示一列为一个样本，所以这里置为False
    covMat = np.cov(meanRemoved, rowvar=False)
    # 求特征值和特征向量
    eigVals, eigVects = np.linalg.eig(covMat)
    eigValInd = np.argsort(eigVals)
    # eigValIndex表示降序排列的topNfeat个特征值
    eigValIndex = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:, eigValIndex]
    # redEigVects (k, n)  meanRemoved (m, n)
    # low represent meanRemoved * regEigVects
    # recover meanRemoved * regEigVects * regEigVects.T   because regEigVects is unitary matrix
    lowDDataMat = meanRemoved * redEigVects
    reconMat = lowDDataMat * redEigVects.T
    return lowDDataMat, reconMat


