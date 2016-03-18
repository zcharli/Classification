import numpy as np
import sklearn.cross_validation as skc
import zoo as z

HEART_DISEASE = "./data/heartDisease.csv"
CPU = "./data/cpu.csv"
ZOO = "./data/zoo.csv"


def main(file):
    trainTestBatch = splitTenFold(loadCSV(file))
    if file == ZOO:
        z.trainZoo(trainTestBatch)

def optimalBayesian(testBatch):
    sample = testBatch[0] # Temporary


def loadCSV(path):
    return np.recfromcsv(path, dtype=float, names=None,delimiter=',')

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

def getClasses(tupleArray, index):
    classes = dict()
    for sample in tupleArray:
        if sample[index] in classes:
            classes[sample[index]].append(sample)
        else:
            classes[sample[index]] = [sample]
    return classes

if __name__ == "__main__":
    main(ZOO)

