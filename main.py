from __future__ import division
import numpy as np
import sklearn.cross_validation as skc
import sklearn.naive_bayes as nb

# Useful settings keys
COVARIANCE = "train_classes_cov_dict"
MEAN = "train_classes_mean_dict"
CLASSES = "train_classes_dict"
HEART_DISEASE = "./data/heartDisease.csv"
CPU = "./data/cpu.csv"
ZOO_SETTING = {"usecols": (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17),
               "data": "./data/zoo.csv",
               "title":"ZOO"}

def main(s):
    trainTestBatch = splitTenFold(loadCSV(s["data"], s))
    s["10_fold_batches"] = trainTestBatch
    if s["title"] == "ZOO":
        trainZoo(s)

def optimalBayesian(testBatch):
    sample = testBatch[0] # Temporary

def loadCSV(path, s):
    return np.genfromtxt(path, dtype=None, usecols=s["usecols"], names=None,delimiter=',')

def splitTenFold(data):
    splitIdx = skc.KFold(len(data), n_folds=10)
    tenFold = []
    for train, test in splitIdx:
        tenFold.append((np.asarray([data[j] for j in train]),
                        np.asarray([data[j] for j in test])))
    return tenFold

def getMeanByClasses(sampleClassDict):
    classMeans = dict()
    for key in sampleClassDict.keys():
        k = sampleClassDict[key]
        m = np.mean(sampleClassDict[key], axis=0)
        classMeans[key] = np.mean(sampleClassDict[key], axis=0)
    return classMeans

def getCovarianceMatrix(sampleClassDict):
    classCov = dict()
    for key in sampleClassDict.keys():
        classCov[key] = np.cov(np.array(sampleClassDict[key]).T)
    return classCov

def trainZoo(zooSettingsDict):
    for batches in zooSettingsDict["10_fold_batches"]:
        trainSample = batches[0]
        testSample = batches[1]
        zooSettingsDict["train_classes_dict"] = getClasses(trainSample)
        zooSettingsDict[MEAN] = getMeanByClasses(zooSettingsDict["train_classes_dict"])
        zooSettingsDict[COVARIANCE] = getCovarianceMatrix(zooSettingsDict["train_classes_dict"])
        #print zooSettingsDict["train_classes_cov_dict"]
        for x in trainSample:
            trueClass = x[len(x) - 1]
            #classsifiedClass =
            classifiedClass = classify(zooSettingsDict, np.asarray(np.delete(x, len(x) - 1)))
            print "e: %d   r: %d" % (trueClass, classifiedClass)

def classify(s, x):
    classes = s[CLASSES].keys()
    if len(classes) < 2: raise ValueError("Cannot classify with only one class")
    decKey = classes[0]
    #minDistance = sys.maxint
    for idx in range(1, len(classes)):
        # a = np.linalg.norm(s[COVARIANCE][decKey])
        # b = np.linalg.norm(s[COVARIANCE][classes[idx]])
        # p = np.log(np.abs(b / a))
        # ln2 = np.log(np.linalg.norm(s[COVARIANCE][classes[idx]]))
        # ln1 = np.log(np.linalg.norm(s[COVARIANCE][decKey]))
        # l3 = np.log( np.abs(p))
        # m2 = mahalanobisDistance(x, s[MEAN][classes[idx]], s[COVARIANCE][classes[idx]])
        # m1 = mahalanobisDistance(x, s[MEAN][decKey], s[COVARIANCE][decKey])
        c = np.log(np.linalg.norm(s[COVARIANCE][classes[idx]])) - np.log(np.linalg.norm(s[COVARIANCE][decKey])) + \
            mahalanobisDistance(x, s[MEAN][classes[idx]], s[COVARIANCE][classes[idx]]) - \
            mahalanobisDistance(x, s[MEAN][decKey], s[COVARIANCE][decKey])
        if c < 0:
            decKey = classes[idx]


    return decKey

def mahalanobisDistance(x, m, c):
    if np.linalg.det(c) == 0.0:
        l = np.linalg.pinv(c)
    else:
        l = np.linalg.inv(c)
    return reduce(np.dot, [np.array(x-m).T, l, x-m])

def getClasses(tupleArray):
    classes = dict()
    if len(tupleArray) == 0: raise ValueError("No classes to get.")
    index = len(tupleArray[0]) - 1
    for sample in tupleArray:
        if sample[index] in classes:
            classes[sample[index]].append(np.asarray(np.delete(sample,len(sample) -1)))
        else:
            classes[sample[index]] = [np.asarray(np.delete(sample,len(sample) -1))]
    return classes

if __name__ == "__main__":
    main(ZOO_SETTING)

# Not needed
# def manualCov(vectorList, mean):
#     c = 0
#     for i in vectorList:
#         x1 = (i - mean)
#         x2 = np.array([i-mean]).T
#         m1 = np.multiply(x1,x2)
#         c += m1
#     return (1/(len(vectorList)-1)) * c
