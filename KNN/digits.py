import numpy as np
import os

def KNN(inX,trainingSet,labels,k:int=10):
    setSize=trainingSet.shape[0]
    diffMat=trainingSet-np.tile(inX,(setSize,1))
    diffMat=diffMat**2
    diffMat=np.sum(diffMat,axis=1)
    diffMat=np.sqrt(diffMat)
    indexs=diffMat.argsort()
    counts={}
    for i in range(k):
        if not labels[indexs[i]] in counts:
            counts[labels[indexs[i]]]=0
        else:
            counts[labels[indexs[i]]]+=1
    counts=sorted(counts.items(),key=lambda x:x[-1],reverse=True)
    return counts[0][0]


if __name__=='__main__':
    # filenames_labels=[]
    # for i in os.listdir('/home/huangchenhan/KNN/ml/Ch02/testDigits'):
    #     filenames_labels.append(('/home/huangchenhan/KNN/ml/Ch02/testDigits/'+i,int(i[0])))
    # print(filenames_labels)
    # testingSet=np.zeros((946,1024),dtype='i1')
    # labels=[]
    # index=0
    # for file,label in filenames_labels:
    #     with open(file,'r') as f:
    #         i=0
    #         for line in f.readlines():
    #             j=0
    #             for digit in line.strip():
    #                 testingSet[index,i*32+j]=digit
    #                 j+=1
    #             i+=1
    #     labels.append(int(label))
    #     index+=1
    # np.savez('testingSet',dataSet=testingSet,labels=labels)
    trainingset=np.load('trainingSet.npz')['dataSet']
    traininglabels=np.load('trainingSet.npz')['labels']

    testingSet,nums=np.load('testingSet.npz')['dataSet'],np.load('testingSet.npz')['dataSet'].shape[0]
    testingLabels=np.load('testingSet.npz')['labels']
    ans=[]
    for i in range(nums):
        ans.append(KNN(testingSet[i],trainingset,traininglabels))
    error=0.0
    for i in range(nums):
        if ans[i]!=testingLabels[i]:
            error+=1.0
    print("{}%".format(round((1-(error/nums))*100,2)))
