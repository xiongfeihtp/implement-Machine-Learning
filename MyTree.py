import numpy as np
from math import log2

from collections import Counter


# 针对某一数据集，计算香农熵
def calShannon(DataSet):
    sample_num = len(DataSet)
    sample_count = Counter()
    for sample in DataSet:
        sample_count.update([sample[-1]])

    ShannonEnt = 0.0
    for label in sample_count:
        prob = float(sample_count[label]) / sample_num
        ShannonEnt -= prob * log2(prob)
    return ShannonEnt


# 数据集划分
def splitDataSet(DataSet, axis, value):
    retDataSet = []
    for featVec in DataSet:
        if (featVec[axis] == value):
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet


# 选择最好的数据集划分方式
def ChooseBestFeatureToSplit(DataSet):
    numFeatures = len(DataSet[0]) - 1
    baseEntropy = calShannon(DataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    bestFeature_value = 0
    for i in range(numFeatures):
        uniqueVals = set([sample[i] for sample in DataSet])
        newEntropy = 0.0
        # 多叉树
        for value in uniqueVals:
            subDataSet = splitDataSet(DataSet, i, value)
            prob = len(subDataSet) / float(len(DataSet))
            newEntropy += prob * calShannon(subDataSet)
        InfoGain = baseEntropy - newEntropy
        if (bestInfoGain < InfoGain):
            bestInfoGain = InfoGain
            bestFeature = i
    return bestFeature


# 投票返回最多的类
def majorityCnt(classList):
    label_count = Counter()
    for vote in classList:
        label_count.update([vote])
    return label_count.most_common(1)[0][0]


# 递归建树过程
def createTree(DataSet, feature_name):
    #当前叶节点的标签情况
    classList = [sample[-1] for sample in DataSet]
    # 叶节点样本标签都一样
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 没有可划分的属性
    if len(DataSet[0]) == 1:
        return majorityCnt(classList)

    #选出最优属性
    bestfeat = ChooseBestFeatureToSplit(DataSet)
    bestfeatname = feature_name[bestfeat]
    myTree = {bestfeatname: {}}
    #删除以选属性
    del feature_name[bestfeat]
    #建立多叉树
    featValues = set([sample[bestfeat] for sample in DataSet])
    for value in featValues:
        myTree[bestfeatname][value] = createTree(splitDataSet(DataSet, bestfeat, value), bestfeatname)

    return myTree



