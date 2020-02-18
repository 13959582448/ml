import numpy as np


def createDataSet():
    postingList = ['my dog had flea problems help please', 'maybe not take him to dog park stupid',
                   'my dalmation is so cute I love him', 'stop posting stupid worthless garbage',
                   'mr licks ate my steak how to stop him', 'quit buying worthless dog food stupid']
    postingList = [i.split() for i in postingList]
    labels = [0, 1, 0, 1, 0, 1]
    return postingList, labels


def createVocabList(dataset) -> list:  # 数据集去重
    vocabList = set([])
    for sentence in dataset:
        vocabList = vocabList | set(sentence)
    return list(vocabList)


def convertFeatureVec(vocabList: int, input) -> list:  # 转换为特征向量
    featureVec = [0] * len(vocabList)
    for word in input:
        if word in vocabList:
            featureVec[vocabList.index(word)] = 1
    return featureVec


def train(trainMat, labels):
    numDocu = len(trainMat)  # 数量
    nump0, nump1 = 2.0, 2.0  # p1 p2类别的词条总数 防止出现除数为0的情况

    p0Mat, p1Mat = np.ones(len(trainMat[0])), np.ones(len(trainMat[0]))  # 每一个词条出现的次数
    for i in range(numDocu):
        if labels[i] == 1:
            p1Mat += trainMat[i]
            nump1 += np.sum(trainMat[i])
        else:
            p0Mat += trainMat[i]
            nump0 += np.sum(trainMat[i])
    p0_probVec = np.log(p0Mat / nump0)  # 代表在类别0 中每个特征出现的概率 P(w(i)/c) 每一个元素代表w(i)在类别c中出现的概率
    p1_probVec = np.log(p1Mat / nump1)  # 代表在类别1 中每个特征出现的概率 // log 用于防止下溢出 (概率过小) 而且乘积可以直接用对数相加表示
    p1_prob = float(np.sum(labels) / numDocu)  # 训练集中类别为1的概率
    p0_prob = 1 - p1_prob
    return p0_probVec, p1_probVec, p0_prob, p1_prob

def classify(featureVec,p0_probVec,p1_probVec,p0_prob,p1_prob):
    p1=sum(p1_probVec*featureVec)+np.log(p1_prob)
    p0=sum(p0_probVec*featureVec)+np.log(p0_prob)
    return 1 if p1>p0 else 0

def testing(testMat):
    dataset,labels=createDataSet()
    vocab=createVocabList(dataset)
    testFeature=convertFeatureVec(vocab,testMat)
    trainingSet=[]
    for i in dataset:
        trainingSet.append(convertFeatureVec(vocab,i))
    p0_probVec,p1_proVec,p0_prob,p1_pro=train(trainingSet,labels)
    result=classify(testFeature,p0_probVec,p1_proVec,p0_prob,p1_pro)
    print(testMat," classified as {}".format(result),sep='')

if __name__ == '__main__':
    testing(np.array("cute ".split()))

