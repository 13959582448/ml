from numpy import *
import numpy as np
import matplotlib.pyplot as plt



if __name__=='__main__':
    file=open(r"dating.txt","r")
    lines=file.readlines()
    dataSet=zeros((len(lines),3),dtype=np.float)
    labels=[]
    lines=[i.strip() for i in lines]
    index=0
    for line in lines:
        line=line.split("\t")
        line[0],line[1],line[2]=float(line[0]),float(line[1]),float(line[2])
        dataSet[index]=line[0:3]
        index+=1
        labels.append(int(line[-1]))

    x=dataSet[:,1]
    y=dataSet[:,2]

    fig=plt.figure()
    ax=fig.add_subplot(223)
    colors=['r','g','b']
    formatcolors=[colors[i-1] for i in labels] #
    ax.scatter(x,y,color=formatcolors,s=10)
    plt.show()

    normalMat=dataSet/np.tile(np.ptp(dataSet,axis=0),(dataSet.shape[0],1)) #
    print(normalMat)


