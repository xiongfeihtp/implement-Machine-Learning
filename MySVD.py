import numpy as np

"""
相似度描述（correlation）
1，欧式距离 dst 对用户的评级比较敏感，更多考虑的局部距离 -> 归一化 1/(1+dst)
2，皮尔逊相关系数， 对用户评级不敏感 (-1, 1) -> 归一化 0.5+0.5*dst
3，余弦相似度和皮尔逊相关系数类似  (-1, 1) ->归一化  0.5+0.5*dst
皮尔逊相关系数就是减去平均值(中心化)后做余弦相似性
"""


def dst1(X, Y):
    return 1.0 / (1.0 + np.linalg.norm(X - Y))


"""
这里使用numpy库的cov函数来计算协方差，输出结果为一个协方差矩阵，
results[i][j]表示第i个变量与第j个变量的协方差,比如np.cov(x)的结果为

x= np.concat(a, b, axis = 0)
np.cov(x) = np.cov(a,b)

np.cov(x)
array([[ 1.        ,  0.        ],
       [ 0.        ,  0.33333333]])

x为变量矩阵，x的第 i 行表示第 i 个随机变量（这里并不是样本，如果是样本特征的排列方式需要rowvar =False），这里第0个随机变量与第0个随机变量的协方差为1，
第0个随机变量与第1个随机变量的协方差为0，第1个随机变量与第0个随机变量的协方差为0，
第1个随机变量与第1个随机变量的协方差为0.33333333.

还有一点就是bais参数的意义，默认为False，那么计算均值的时候除以 n - 1，如果设为True，（无偏估计）
那么计算均值的时候除以n，其中n为随机变量的维度数。具体理论涉及到统计学知识，可以参考知乎回答。
"""


def dst2(X, Y):
    # np.corrcoef返回的是一个矩阵
    return 0.5 + 0.5 * np.corrcoef(X, Y, rowvar=False)[0][1]


def dst3(X, Y):
    num = float(X.T * Y)
    denom = np.linalg.norm(X) * np.linalg.norm(Y)
    return 0.5 + 0.5 * (num / denom)



#针对一个用户，返回用户对item的可能评分，基于物品进行预测，通常的做法，因为用户数量一般比物品数目多的多
def svdEnt(dataMat, user, simMeas, item):
    """
    np.linalg.svd() 分解输入的最后两个维度
    input: M*N
    U: M*M  sigma: min(M,N) VT: N*N
    """
    U, sigma, VT = np.linalg.svd(dataMat)
    """
    Sig4 = np.mat(np.eye(4) * sigma[:4])  #转换为奇异值组成的对角阵
    UserTransfer = dataMat.T * U[:,:4] * Sig4.I  #多次一举
    """
    ItemTransfer = VT[:4, :].T
    # 找出和user相关的所有item
    user_rating = dataMat[user, :]  #根据用户中item的出现次数对item进行加权
    simList = []
    for _item in dataMat[user, :]:
        if _item == 0 or _item == item:
            simList.append(0)
        else:
            simList.append(simMeas(ItemTransfer[_item, :], ItemTransfer[item, :]))
    allRate = np.array(simList) * np.array(user_rating).sum()
    totalSimilarity = sum(simList)
    #返回对用户针对某一个物品的预测
    return allRate/totalSimilarity



