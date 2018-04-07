import numpy as np

import tensorflow as tf

tf.nn.batch_normalization()


# 切分数据集
def binSplitDataSet(DataSet, feature, value):
    mat0 = DataSet[np.nonzero(DataSet[:, feature] > value)[0], :]
    mat1 = DataSet[np.nonzero(DataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 切分函数
# 计算叶节点的最优增益
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


# 计算当前数据集均方差
def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 设置这两个参数，涉及预剪枝
    tolS = ops[0]  # 容许的误差下降值
    to1N = ops[1]  # 容许的分裂最小值
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestvalue = 0
    for featindex in range(n - 1):
        for splitVal in set(dataSet[:, featindex]):
            mat0, mat1 = binSplitDataSet(dataSet, featindex, splitVal)
            if (np.shape(mat0)[0] < to1N) or (np.shape(mat1)[0] < to1N):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featindex
                bestvalue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestvalue)
    if (np.shape(mat0)[0] < to1N) or (np.shape(mat1)[0] < to1N):
        return None, leafType(dataSet)
    return bestIndex, bestvalue


# 建树函数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 找到最优的分割属性和叶节点值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 递归建树
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 后剪枝（通过测试集数据来实现）
def isTree(obj):
    return (type(obj)._name__ == 'dict')


# 求树的平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0: return getMean(tree)
    if isTree(tree['left']) or isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    # 当检测到两个叶节点时进行合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        # 递归到两个叶节点，判断是否需要进行合并
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum((lSet[:, -1] - tree['left']) ** 2) + np.sum((rSet[:, -1] - tree['right']) ** 2)
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum((testData[:, -1] - treeMean) ** 2)
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

# CART树的扩展，前面叶节点只是代表一个固定的值，这里改成特定的模型，构成模型树，可以用于分类等问题，
# 只需修改传入的节点类型函数和误差计算函数就可以
def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError("this matrix is singular")
    ws = xTx * (X.T * Y)  # 矩阵相乘通过括号进行计算量优化
    return ws, X, Y

def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum((Y - yHat) ** 2)




