import numpy as np
import csv
import sklearn.cross_validation as skc
import zoo as z

HEART_DISEASE = "./data/heartDisease.csv"
CPU = "./data/cpu.csv"
ZOO = "./data/zoo.csv"
ZOO_SETTING = {"usecols": (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)}

def main(data, s):
    trainTestBatch = splitTenFold(loadCSV(data, s))
    if data == ZOO:
        z.trainZoo(trainTestBatch)

def optimalBayesian(testBatch):
    sample = testBatch[0] # Temporary


def loadCSV(path, s):
    return np.genfromtxt(path, dtype=None, usecols=s["usecols"], names=None,delimiter=',')

def splitTenFold(data):
    splitIdx = skc.KFold(len(data), n_folds=10)
    tenFold = []
    for train, test in splitIdx:
        tenFold.append(([data[j] for j in train],
                        [data[j] for j in test]))
    return tenFold

def getMeanByClasses(sampleClassDict):
    classMeans = dict()
    for key,v in sampleClassDict.iteritems():
        classMeans[key] = np.mean(v, axis=0)
    return classMeans

def getCovarianceMatrix(nd):
    return np.cov(nd)

def getClasses(tupleArray, index):
    classes = dict()
    for sample in tupleArray:
        if sample[index] in classes:
            classes[sample[index]].append(sample)
        else:
            classes[sample[index]] = [sample]
    return classes

if __name__ == "__main__":
    main(ZOO, ZOO_SETTING)

