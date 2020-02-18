# 决策树
import numpy as np
import matplotlib.pyplot as plt
import random
from math import log


def createDataSet():
    dataSet = [['yes', 'yes', 'yes'],
               ['yes', 'yes', 'yes'],
               ['yes', 'no', 'no'],
               ['no', 'yes', 'no'],
               ['no', 'yes', 'no']]
    labels = ['No surfacing', 'Flippers']
    return dataSet, labels


def calShannonEnt(dataset):
    setSize = len(dataset)
    labelCount = {}
    for featureVec in dataset:
        label = featureVec[-1]
        if label not in labelCount:
            labelCount[label] = 0
        labelCount[label] += 1
    shannonEnt = 0.0
    for key, value in labelCount.items():
        prob = float(value / setSize)
        shannonEnt -= prob * (log(prob, 2))  # -log2p(x)
    return shannonEnt


def splitDataset(dataset, axis, value):
    returnMat = []
    for featureVec in dataset:
        if featureVec[axis] == value:
            reducedFeatureVec = featureVec[:axis]
            reducedFeatureVec.extend(featureVec[axis + 1:])  # 除去那一个特征值，因为作为划分的标准了
            returnMat.append(reducedFeatureVec)
    return returnMat


def chooseBestFeatureToSplit(dataset):  # 选出最适合的一个特征，但没有选出这个特征的具体划分的值
    baseEnt = calShannonEnt(dataset)  # 划分前的信息熵
    numsFeature = len(dataset[0]) - 1  # 特征数量
    bestFeature = -1  # 最佳划分的特征
    infoGain = 0.0  # 信息增益
    for axis in range(numsFeature):
        values = [i[axis] for i in dataset] #一列特征值
        values = set(values)  # 去重
        newEnt = 0.0
        for value in values:  # 一个维度的按每一个值划分情况 所有子集都不想相互包含 用n个节点的树 不一定是二叉树
            subDataSet = splitDataset(dataset, axis, value)
            prob = len(subDataSet) / float(len(dataset))
            newEnt += prob * calShannonEnt(subDataSet)  # 求期望
        if baseEnt - newEnt > infoGain:
            infoGain = baseEnt - newEnt
            bestFeature = axis
    return bestFeature


def majorityCnt(classList):  # 返回列表中出现最多的值
    counts = {}
    for i in classList:
        if i not in counts:
            counts[i] = 0
        counts[i] += 1
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return counts[0][0]


def createTree(dataSet, labels):  # label指特征类
    classList = [i[-1] for i in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel: {}}
    del (labels[bestFeature])
    featureValues = [example[bestFeature] for example in dataSet]
    featureValues = set(featureValues)
    for value in featureValues:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataset(dataSet, bestFeature, value), subLabels)
    return myTree


if __name__ == '__main__':
    dataset, labels = createDataSet()
    print(createTree(dataset, labels))
