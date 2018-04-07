import numpy as np


# 学习矩阵形式的计算
# 文档对于词的one-hot表示矩阵
# 这里针对的是二分类问题
def trainNB(trainMatrix, trainCategory):
    sample_nums = len(trainMatrix)
    num_word = len(trainMatrix[0])
    #记录p(ci)
    pAbusive = sum(trainCategory) / float(sample_nums)

    # 记录p(w|ci)
    # 进行laplace修正
    p0Num = np.ones(num_word)
    p1Num = np.ones(num_word)
    p0 = 2.0
    p1 = 2.0

    for i in range(sample_nums):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1 += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0 += sum(trainMatrix[i])
    #极大似然
    p1Vec = np.log(p1Num / p1)  #保存为log
    p0VeC = np.log(p0Num / p0)  #保存为log

    return p0Num, p1Num, pAbusive



