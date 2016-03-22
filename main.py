from __future__ import division
import numpy as np
import sklearn.cross_validation as skc
import sys

# Ignore those annoying errors
# np.seterr(invalid='ignore',divide='ignore')

# Enums
LINEAR = 0
OPTIMAL_BAYES = 1
NAIVE_BAYES = 2

# Useful settings keys
COVARIANCE = "train_classes_cov_dict"
MEAN = "train_classes_mean_dict"
CLASSES = "train_classes_dict"

IRIS_SETTING = {"usecols": (0,1,2,3,4), "classIndex":4,
               "data": "./data/iris.csv",
               "title":"ZOO"}
WINE_SETTING = {"usecols": (0,1,2,3,4,5,6,7,8,9,10,11,12,13),"classIndex":0,
               "data": "./data/wine.csv",
               "title":"WINE"}
HEART_SETTING = {"usecols": (0,2,3,4,5,6,7,8,9,10,11,12,13),"classIndex":12,
               "data": "./data/heartDisease.csv",
               "title":"HEARTDISEASE"}


def main(s, strat):
    trainTestBatch = splitTenFold(loadCSV(s["data"], s), s)
    s["10_fold_batches"] = trainTestBatch
    train(s, strat)

def loadCSV(path, s):
    return np.genfromtxt(path, dtype=None, usecols=s["usecols"], names=True, delimiter=',')

def processTuple(t,s):
    if t[s['classIndex']].dtype.type is np.string_:
        l = [t[x] for x in xrange(s['classIndex']+1)]# + t[s['classIndex']]
        return np.asarray(l, dtype=object)
    return np.asarray(np.asarray(t,dtype=object).item(0))

def splitTenFold(data, s):
    splitIdx = skc.KFold(len(data), n_folds=10) #, shuffle=True)
    tenFold = []
    for train, test in splitIdx:
        tenFold.append(([processTuple(data[j], s) for j in train],
                        [processTuple(data[j], s) for j in test]))
    return tenFold

def getMeanByClasses(sampleClassDict):
    classMeans = dict()
    for key in sampleClassDict.keys():
        classMeans[key] = np.mean(sampleClassDict[key], axis=0)
    return classMeans

def getCovarianceMatrix(sampleClassDict, strat):
    classCov = dict()
    if strat == LINEAR:
        return 0
    else:
        for key in sampleClassDict.keys():
            classCov[key] = np.cov(np.array(sampleClassDict[key]).T)
            if strat == NAIVE_BAYES:
                classCov[key] = np.diag(np.diag(classCov[key]))
    return classCov


def getClasses(tupleArray, i):
    classes = dict()
    if len(tupleArray) == 0: raise ValueError("No classes to get.")
    for sample in tupleArray:
        sampleClass = sample[i]
        if sampleClass in classes:
            classes[sampleClass].append(np.delete(sample, i))
        else:
            classes[sampleClass] = [np.delete(sample, i)]
    return classes

def train(s, strat):
    avgCorrect = 0
    for batches in s["10_fold_batches"]:
        trainSample, testSample = batches[0], batches[1]
        s[CLASSES] = getClasses(trainSample, s["classIndex"])
        s[MEAN] = getMeanByClasses(s[CLASSES])
        s[COVARIANCE] = getCovarianceMatrix(s[CLASSES], strat)
        batchCorrect,batchTotal = 0, len(trainSample)
        for x in trainSample:
            trueClass = x[s['classIndex']]
            classifiedClass = classify(s, np.delete(x,s['classIndex']), strat)
            batchCorrect = batchCorrect + 1 if trueClass == classifiedClass else batchCorrect
        avgCorrect += (batchCorrect/batchTotal)
    print "10-Fold accuracy for %s dataset is %.2f" % (s["title"], avgCorrect/10)

def classify(s, x, strat):
    classes = s[CLASSES].keys()
    if len(classes) < 2: raise ValueError("Cannot classify with only one class")
    decKey = classes[0]
    for idx in range(1, len(classes)):
        if strat == LINEAR:
            c = 0
        else:
            c = getLn(s[COVARIANCE][classes[idx]]) - getLn(s[COVARIANCE][decKey]) + \
                mahalanobisDistance(x, s[MEAN][classes[idx]], s[COVARIANCE][classes[idx]]) - \
                mahalanobisDistance(x, s[MEAN][decKey], s[COVARIANCE][decKey])
        if c < 0:
            decKey = classes[idx]
    return decKey

def getLn(covarianceMatrix):
    if np.linalg.cond(covarianceMatrix) < 1/sys.float_info.epsilon:
        return np.log(np.linalg.det(covarianceMatrix))
    else:
        d = np.linalg.det(covarianceMatrix)
        if d == 0: return 0
        return np.log(d) if d != 0 else d

def mahalanobisDistance(x, m, c):
    if np.linalg.det(c) == 0.0:
        l = np.linalg.pinv(c)
    else:
        l = np.linalg.inv(c)
    return reduce(np.dot, [np.array(x-m).T, l, x-m])

if __name__ == "__main__":
   main(WINE_SETTING, OPTIMAL_BAYES)
   main(IRIS_SETTING, OPTIMAL_BAYES)
   main(HEART_SETTING, OPTIMAL_BAYES)
   main(HEART_SETTING, NAIVE_BAYES)
   main(IRIS_SETTING, NAIVE_BAYES)
   main(WINE_SETTING, NAIVE_BAYES)

# Not needed
# def manualCov(vectorList, mean):
#     c = 0
#     for i in vectorList:
#         x1 = (i - mean)
#         x2 = np.array([i-mean]).T
#         m1 = np.multiply(x1,x2)
#         c += m1
#     return (1/(len(vectorList)-1)) * c
