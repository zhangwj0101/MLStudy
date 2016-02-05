import numpy as np
import matplotlib.pyplot as plt

path = '../../data/'


def loadData():
    datas = np.loadtxt(path + 'testSet.txt', delimiter='\t')
    m, n = np.shape(datas)
    s = np.ones((m, 1), dtype=float)
    retData = np.hstack((s, datas))
    return retData[:, :3], retData[:, 3].reshape(m, 1)


def loadhourseData(fileName):
    retData = np.loadtxt(fileName, delimiter='\t')
    m, n = np.shape(retData)
    return retData[:, :n - 1], retData[:, n - 1].reshape(m, 1)


# DataMat,DataLabel = loadData()
hoursetrainDataMat, hoursetrainDataLabel = loadhourseData(path + 'horseColicTraining.txt')
hoursetestDataMat, hoursetestDataLabel = loadhourseData(path + 'horseColicTest.txt')


# print hoursetrainDataMat,hoursetrainDataLabel

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def gradWeight(DataMat, DataLabel, iterTime=300, alpha=0.01):
    m, n = np.shape(DataMat)
    weights = np.ones((1, n))
    difs = weights

    for i in range(iterTime):
        dif = DataLabel - sigmoid(DataMat * weights.T)
        weights = weights + alpha * dif.T * DataMat
        difs = np.vstack((difs, weights))
    return [(weights, difs)]


def randgradWeight(DataMat, DataLabel, alpha=0.01):
    m, n = np.shape(DataMat)
    weights = np.ones((1, n))
    difs = weights

    for i in range(m):
        dif = DataLabel[i] - sigmoid((DataMat[i] * weights.T)[0])
        weights = weights + alpha * dif * DataMat[i]
        difs = np.vstack((difs, weights))
    return [(weights, difs)]


def randgradWeight1(DataMat, DataLabel, iterTime=150, alpha=0.01):
    m, n = np.shape(DataMat)
    weights = np.ones((1, n))
    difs = weights
    for j in range(iterTime):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4. / (1. + j + j) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            dif = DataLabel[i] - sigmoid((DataMat[randIndex] * weights.T)[0])
            weights = weights + alpha * dif * DataMat[randIndex]
            difs = np.vstack((difs, weights))
            del (dataIndex[randIndex])
    return [(weights, difs)]


# weights = gradWeight(np.mat(DataMat), np.mat(DataLabel),iterTime=400)
# randweights = randgradWeight(np.mat(DataMat), np.mat(DataLabel))
# rand1weights = randgradWeight1(np.mat(DataMat), np.mat(DataLabel))

def logShow(DataMat, DataLabel, *weights):
    indices1 = np.where(DataLabel == 1)
    indices0 = np.where(DataLabel == 0)
    m, n = np.shape(DataMat)
    l = len(weights)
    plt.figure(figsize=(12, 9))
    g = plt.GridSpec(l * 3, 2)
    i = 0;
    j = 0
    for w in weights:
        plt.subplot(g[i:i + 3, 0])
        i += 3
        plt.scatter(DataMat[indices1, 1], DataMat[indices1, 2], c='r', s=60)
        plt.scatter(DataMat[indices0, 1], DataMat[indices0, 2], s=90)
        weight = w[0][0]
        y = -(weight[0, 0] + weight[0, 1] * DataMat[:, 1]) / weight[0, 2]
        plt.plot(DataMat[:, 1], y)

        difs = w[0][1]
        x = range(len(difs))
        plt.subplot(g[j, 1])
        plt.plot(x, difs[:, 0].flatten().A[0], c='r')
        plt.subplot(g[j + 1, 1])
        plt.plot(x, difs[:, 1].flatten().A[0], c='g')
        plt.subplot(g[j + 2, 1])
        plt.plot(x, difs[:, 2].flatten().A[0], c='b')
        j += 3

    plt.show()


def test(hoursetestDataMat, hoursetestDataLabel, weights):
    m, n = np.shape(hoursetestDataMat)
    rightCount = 0
    errorCount = 0
    # print weights.T
    for i in range(m):
        p = sigmoid(np.sum(hoursetestDataMat[i] * weights.T))
        if p > 0.5:
            label = 1.
        else:
            label = 0.
        # print p
        if label == hoursetestDataLabel[i]:
            print 'line %d is right pro is  %f' % (i + 1, p)
            rightCount += 1
        else:
            print 'line %d is wrong pro is  %f-------' % (i + 1, p)
            errorCount += 1
    print 'right is', rightCount
    print 'error is', errorCount
    print 'rates : ', float(rightCount) / (rightCount + errorCount)
    return float(rightCount) / (rightCount + errorCount)


def hourseShow():
    iterTime = np.arange(100, 550, 10)
    print iterTime
    rateArr = []
    highRate = 0.
    lowRate = 1.
    hoursetrainDataMat, hoursetrainDataLabel = loadhourseData(path + 'horseColicTraining.txt')
    hoursetestDataMat, hoursetestDataLabel = loadhourseData(path + 'horseColicTest.txt')
    for i in iterTime:
        weights = gradWeight(np.mat(hoursetrainDataMat), np.mat(hoursetrainDataLabel), iterTime=i)
        rates = test(hoursetestDataMat, hoursetestDataLabel, weights[0][0])
        print rates
        if (highRate < rates):
            highRate = rates
            highWeights = weights
        if (lowRate > rates):
            lowRate = rates
            lowWeights = weights
        rateArr.append(rates)
    hourselogShow(iterTime, rateArr, highWeights, lowWeights)


def hourselogShow(iterTime, rateArr, *weights):
    l = len(weights)
    plt.figure(figsize=(20, 9))
    g = plt.GridSpec(l, np.shape(weights[0][0][0])[1])
    i = 0;
    j = 0;
    k = 0
    col = ['r', 'g', 'b']
    for w in weights:
        wei = w[0][1]
        print wei.shape
        x = range(len(wei))
        for j in range(np.shape(wei)[1]):
            plt.subplot(g[i, j])
            plt.plot(x, wei[:, j].flatten().A[0], col[k])
            k = k % len(col)
        i += 1

    plt.show()


print "s"
hourseShow()
