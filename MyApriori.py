from optparse import OptionParser
from tqdm import tqdm
from collections import defaultdict
from itertools import chain, combinations
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


# 文件流生成
def DataFromFile(file):
    file_iter = open(file, 'rU')
    for line in file_iter:
        line = line.strip().replace('"', '')
        record = line.split(',')[0:10]
        yield record


def ParserFileStream(FileStream):
    print("begin parsering file")
    SampleList = list()
    ItemSet = set()
    # 因为后期需要作为key值进行查询，所以建立为frozenset
    for sample in tqdm(FileStream):
        coms = frozenset(sample)
        SampleList.append(coms)
        for item in coms:
            ItemSet.add(frozenset([item]))
    return ItemSet, SampleList


def ItemsFilter(ItemSet, SampleList, minSupport, freqSet):
    # 创建临时容器进行统计
    SampleNums = len(SampleList)
    _itemSet = set()
    localdict = defaultdict(int)

    # 这里速度很慢，多线程进行优化
    # 改成多线程
    def one_step(item):
        for sample in SampleList:
            if item.issubset(sample):
                freqSet[item] += 1
                localdict[item] += 1

    with ThreadPoolExecutor(2) as executor:
        for i, item in enumerate(ItemSet):
            if (i % 100 == 0):
                print(i)
            executor.submit(one_step, item)
    # for item in tqdm(ItemSet):
    #     one_step(item)
    for item, count in localdict.items():
        support = count / SampleNums
        if support >= minSupport:
            _itemSet.add(item)
    return _itemSet


def joinSet(ItemSet, Length):
    return set([i.union(j) for i in ItemSet for j in ItemSet if len(i.union(j)) == Length])


def subSets(items):
    return chain(*[combinations(items, i) for i in range(1, len(items))])

def runApriori(FileStream, minSupport, minConfidence):
    # 解析文件流
    ItemSet, SampleList = ParserFileStream(FileStream)
    # 维护两个全局结构，保存所有的item freq和每个长度的满足支持度的item集合
    freqSet = defaultdict(int)
    LargeDict = dict()

    # 进行筛选：
    print("filtering: 1 item number: {}".format(len(ItemSet)))
    ItemSet_r1 = ItemsFilter(ItemSet, SampleList, minSupport, freqSet)
    BeforeSet = ItemSet_r1
    k = 2
    while (BeforeSet != set([])):
        LargeDict[k - 1] = BeforeSet
        # 组合满足条件的items，生成下一层的items
        AfterSet = joinSet(BeforeSet, k)
        print("filtering: {} item number: {}".format(k, len(AfterSet)))
        BeforeSet = ItemsFilter(AfterSet, SampleList, minSupport, freqSet)
        k += 1

    def GetSupport(item):
        return float(freqSet[item]) / len(SampleList)

    SupportList = []
    for length, miniSet in LargeDict.items():
        SupportList.extend([(tuple(item), GetSupport(item)) for item in miniSet])

    ConfidenceList = []
    for length, miniSet in tqdm(list(LargeDict.items())[1:]):
        for items in miniSet:
            # 因为后面要进行索引，所以也批量转换为frozenset
            _subset = map(frozenset, [item for item in subSets(items)])
            for before_item in _subset:
                after_item = items.difference(before_item)
                confidence = GetSupport(items) / GetSupport(before_item)
                if confidence >= minConfidence:
                    ConfidenceList.append(((before_item, after_item), confidence))

    return SupportList, ConfidenceList


def OutPrint(SupportList, ConfidenceList):
    SupportResult = open('SupportResult.txt', 'w')
    for item, support in sorted(SupportList, key=lambda x: x[1], reverse=True):
        print("item: {}, {}".format(' '.join(item), support), file=SupportResult)
    SupportResult.close()
    ConfidenceResult = open('ConfidenceResult.txt', 'w')
    for item, confidence in sorted(ConfidenceList, key=lambda x: x[1], reverse=True):
        print("{} -> {},{}".format(' '.join(item[0]), ' '.join(item[1]), confidence), file=ConfidenceResult)
    ConfidenceResult.close()


if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv',
                         default="./data_association.csv")

    optparser.add_option('-s', '--minSupport',
                         dest='minS',
                         help='minimum support value',
                         default=0.0,
                         type='float')
    optparser.add_option('-c', '--minConfidence',
                         dest='minC',
                         help='minimum confidence value',
                         default=0.0,
                         type='float')

    (options, args) = optparser.parse_args()
    FileStream = DataFromFile(options.input)
    minSupport = options.minS
    minConfidence = options.minC
    SupportList, ConfidenceList = runApriori(FileStream, minSupport, minConfidence)
    OutPrint(SupportList, ConfidenceList)
