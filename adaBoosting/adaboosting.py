import numpy as np
from numpy import *
from math import *
import matplotlib.pyplot as plt


def loaddataMat():
    dataMat = mat(
        [[1., 2.1],
         [2., 1.1],
         [1.3, 1.],
         [1., 1.],
         [2., 1.]]
    )
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, labels


def stumpClassify(dataMat, axis, threshVal, ineq) -> np.mat:
    retArray = ones((dataMat.shape[0], 1))
    if ineq == 'lt':
        retArray[dataMat[:, axis] <= threshVal] = -1.0
    else:
        retArray[dataMat[:, axis] > threshVal] = -1.0
    return retArray


def buildStump(dataMat, labels, D) -> [dict, float, np.array]:
    # labels = mat(labels).T
    step = 10.0
    minError = np.inf
    bestStump = {}
    bestClassEst = mat(zeros((dataMat.shape[0], 1)))
    for axis in range(dataMat.shape[1]):  #
        minVal = dataMat[:, axis].min()  #
        maxVal = dataMat[:, axis].max()
        stepSize = (maxVal - minVal) / step
        for j in range(-1, int(step) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = minVal + float(j) * stepSize
                predictedArray = stumpClassify(dataMat, axis, threshVal, inequal)
                errorMat = mat(ones((predictedArray.shape[0], 1)))
                errorMat[predictedArray == labels] = 0
                error = float(D.T * errorMat)
                # print("dim:{} thres:{} ineq:{} error:{}".format(axis, round(threshVal, 1), inequal, round(error, 2)))
                if error < minError:
                    minError = error
                    bestStump['dim'] = j
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
                    bestClassEst = predictedArray.copy()
    return bestStump, minError, bestClassEst


def addBoostTrainDS(dataMat, classLabels, numIt=95):
    show = False

    weakerClassfiers = []
    size = dataMat.shape[0]
    D = mat(ones((size, 1)) / size)
    aggClassEst = mat(zeros((size, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataMat, classLabels, D)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakerClassfiers.append(bestStump)

        if show: print("bestStump:{}\nerror:{}".format(bestStump, error))

        expon = multiply(-1.0 * alpha * mat(classLabels), classEst)
        D = multiply(D, np.exp(expon))
        D = D / D.sum()  # 计算权重

        aggClassEst += alpha * classEst  # 目前为止的总体预测结果
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels), ones((size, 1)))  # 错误
        errorRate = np.sum(aggErrors) / size  # 错误率

        if show: print("errorRate:{}\n\n".format(errorRate))
        if errorRate == 0.0:
            break
    return weakerClassfiers


def adaClassify(dataMat, classifiers):
    dataMat = mat(dataMat)
    m = dataMat.shape[0]
    aggClassEst = mat(zeros((m, 1)))
    for classfier in classifiers:
        classEst = stumpClassify(dataMat, classfier['dim'], classfier['thresh'], classfier['ineq'])
        aggClassEst += classfier['alpha'] * classEst
    return sign(aggClassEst)


def loadDataSet():
    testingFile = open('/home/huangchenhan/ml/Ch07/horseColicTest2.txt', 'r')
    trainingFile = open('/home/huangchenhan/ml/Ch07/horseColicTraining2.txt', 'r')
    trainingMat = [line.strip().split('\t') for line in trainingFile.readlines()]
    testingMat = [line.strip().split('\t') for line in testingFile.readlines()]
    trainingLabels = [float(i[-1]) for i in trainingMat]
    trainingMat = [[float(i[j]) for j in range(len(i) - 1)] for i in trainingMat]
    testingLabels = [float(i[-1]) for i in testingMat]
    testingMat = [[float(i[j]) for j in range(len(i) - 1)] for i in testingMat]
    return mat(trainingMat), mat(trainingLabels).T, mat(testingMat), mat(testingLabels).T


def test(testingMat, testingLabels, classfiers) -> float:
    m = testingMat.shape[0]
    aggClassEst = mat(zeros((m, 1)))
    for classifier in classfiers:
        classEst = stumpClassify(testingMat, classifier['dim'], classifier['thresh'], classifier['ineq'])
        aggClassEst += multiply(classifier['alpha'], classEst)

    errorMat = mat(zeros((m, 1)))
    errorMat[sign(aggClassEst) != testingLabels] = 1

    errorRate = errorMat.sum() / m
    print(errorRate)
    return errorRate


if __name__ == '__main__':
    dataSet = load('dataSet.npz')
    trainingMat, trainingLabels, testingMat, testingLabels = dataSet['trainingMat'], dataSet['trainingLabels'], dataSet[
        'testingMat'], dataSet['testingLabels']

    classifiers = addBoostTrainDS(trainingMat, trainingLabels)
    test(testingMat, testingLabels, classifiers)
    # print(trainingMat, trainingLabels, testingMat, testingLabels, sep='\n')
    #
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    #
    # colors=['r','g']
    # formatcolors=[colors[int(i>0)] for i in labels]
    #
    # shape=['o','v']
    # formatshapes=[shape[int(i>0)] for i in labels]
    # ax.scatter(np.array(dataMat)[...,0],np.array(dataMat)[...,1],color=formatcolors,marker='v')
    #
    # plt.show()
