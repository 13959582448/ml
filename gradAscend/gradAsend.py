import numpy as np
import matplotlib.pyplot as plt
import random


def loadDataset():
    dataMat = []
    labelMat = []
    with open(r'/home/huangchenhan/ml/Ch05/testSet.txt', 'r') as fr:
        for line in fr.readlines():
            line = line.strip().split()
            dataMat.append([1.0, float(line[0]), float(line[1])])
            labelMat.append(int(line[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAsend(dataset, labels):
    dataMat = np.mat(dataset)
    labelMat = np.mat(labels).transpose()
    alpha = 0.001
    cycles = 500
    weights = np.ones((len(dataset[0]), 1))
    for i in range(cycles):
        h = sigmoid(dataMat * weights)
        cost = labelMat - h
        weights = weights + alpha * dataMat.transpose() * cost
    return weights


def randomGradAsend(dataset, labels, cycles=200):
    dataMat = np.mat(dataset)
    labelMat = np.mat(labels).transpose()

    weights = np.ones((dataMat.shape)[1])
    weights = np.mat(weights)
    for j in range(cycles):
        dataIndex = list(range(dataMat.shape[0]))
        for i in range(len(dataset)):
            alpha = 1.0 / (i + j + 1) + 0.01
            randomIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(np.sum(dataMat[randomIndex] * (weights.T)))
            cost = np.sum(labelMat[randomIndex] - h)
            weights = weights + alpha * dataMat[randomIndex] * cost
    return weights


if __name__ == '__main__':
    dataset, labels = loadDataset()
    weights = randomGradAsend(dataset, labels)
    weights = np.array(weights)[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = ['r', 'b']
    formatcolor = [color[i] for i in labels]
    ax.scatter(np.array(dataset)[..., 1], np.array(dataset)[..., 2], color=formatcolor)

    x = np.arange(-3, 3, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y, linewidth=1)
    plt.show()
    print(weights)
