import numpy as np
from collections import Counter

# KNN对特征值的范围敏感，一般采用需要进行归一化
def autoNorm(DataSet):
    minVals = DataSet.min(0)
    maxVals = DataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(DataSet.shape)
    n = DataSet.shape[0]
    normDataSet = DataSet - np.tile(minVals, (n, 1))
    normDataSet = normDataSet / np.tile(ranges, (n, 1))
    del DataSet
    return normDataSet, minVals, maxVals


def KNN(point, DataSet, labels, k):
    sample_num = DataSet.shape
    # 针对矩阵的一些操作，避免进行for循环
    diffMat = np.tile(point, (sample_num, 1)) - DataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sorted_index = distances.argsort()
    vote_count = Counter()
    for index in sorted_index[:k]:
        vote_count.update([labels[index]])
    return vote_count.most_common(1)[0][0]
