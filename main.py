from __future__ import division
import numpy as np
import sklearn.cross_validation as skc
import sys

# Ignore those annoying errors
# np.seterr(invalid='ignore',divide='ignore')

# Enums
LINEAR, OPTIMAL_BAYES, NAIVE_BAYES = "LINEAR", "OPTIMAL BAYES", "NAIVE BAYES"
# Useful settings keys
COVARIANCE, MEAN, CLASS_INDEX, TITLE\
    = "train_classes_cov", "train_classes_mean","classIndex","title"
TEN_FOLD, T_CLASS, TEST_BATCH, TRAIN, TEST = "10_fold_batches", "10_fold_classes", "test_data", 0, 1

IRIS_SETTING = {"usecols": (0,1,2,3,4), "classIndex":4,
               "data": "./data/iris.csv",
               "title":"ZOO"}
WINE_SETTING = {"usecols": (0,1,2,3,4,5,6,7,8,9,10,11,12,13),"classIndex":0,
               "data": "./data/wine.csv",
               "title":"WINE"}
HEART_SETTING = {"usecols": (0,2,3,4,6,7,9,10,11,12,13),"classIndex":10,
               "data": "./data/heartDisease.csv",
               "title":"HEARTDISEASE"}


def main(s, strat):
    splitTenFold(loadCSV(s["data"], s), s)
    train(s, strat)
    test(s, strat)

def loadCSV(path, s):
    return np.genfromtxt(path, dtype=None, usecols=s["usecols"], names=True, delimiter=',')

def processTuple(t,s):
    if t[s[CLASS_INDEX]].dtype.type is np.string_:
        l = [t[x] for x in xrange(s['classIndex']+1)]
        return np.asarray(l, dtype=object)
    return np.asarray(np.asarray(t,dtype=object).item(0))

def splitTenFold(data, s):
    classes, s[T_CLASS] = getClasses(data, s), dict()
    for key in classes.keys():
        splitIdx = skc.KFold(len(classes[key]), n_folds=10, shuffle=True)
        s[T_CLASS][key] = dict()
        s[T_CLASS][key][TEST_BATCH] = []
        for train, test in splitIdx:
            s[T_CLASS][key][TEST_BATCH].append(([classes[key][j] for j in train],
                                           [classes[key][j] for j in test]))

def getCovarianceMatrix(dataset, strat):
    if strat == LINEAR:
        return 0
    else:
        return np.cov(np.array(dataset).T) if strat == OPTIMAL_BAYES else np.diag(np.diag(np.cov(np.array(dataset).T)))

def getClasses(tupleArray, s):
    classes, i = dict(), s[CLASS_INDEX]
    if len(tupleArray) == 0: raise ValueError("No classes to get.")
    for sample in tupleArray:
        sampleClass, sampleFeatures = sample[i], processTuple(sample,s)
        if sampleClass in classes:
            classes[sampleClass].append(np.delete(sampleFeatures, i))
        else:
            classes[sampleClass] = [np.delete(sampleFeatures, i)]
    return classes

def test(s, strat):
    totalCorrectness = 0
    for key in s[T_CLASS].keys():
        classCorrectness = 0
        for batchNumber in xrange(len(s[T_CLASS][key])):
            testSample = s[T_CLASS][key][TEST_BATCH][batchNumber][TEST]
            classCorrect, classTotal = 0, len(testSample)
            for x in testSample:
                classifiedClass = classify(s, x, strat, batchNumber)
                classCorrect = classCorrect + 1 if key == classifiedClass else classCorrect
            classCorrectness += (classCorrect/classTotal)
        print "%s class %s for %s accuracy %0.2f%%" % (s[TITLE], str(key), strat,
                                                       (classCorrectness/len(s[T_CLASS].keys()))*100)
        totalCorrectness += classCorrectness
    print "10-Fold accuracy using %s for %s dataset is %.2f%%\n" % (strat, s[TITLE],
                                                                100*(totalCorrectness/(len(s[T_CLASS].keys())**2)))

def train(s, strat):
    for key in s[T_CLASS].keys():
        s[T_CLASS][key][MEAN], s[T_CLASS][key][COVARIANCE] = [], []
        for i in xrange(len(s[T_CLASS][key][TEST_BATCH])):
            trainSample = s[T_CLASS][key][TEST_BATCH][i][TRAIN]
            s[T_CLASS][key][MEAN].append(np.mean(trainSample, axis=0))
            s[T_CLASS][key][COVARIANCE].append(getCovarianceMatrix(trainSample, strat))

def classify(s, x, strat, b):
    classes = s[T_CLASS].keys()
    if len(classes) < 2: raise ValueError("Cannot classify with only one class")
    decKey = classes[0]
    for idx in range(1, len(classes)):
        if strat == LINEAR:
            c = 0
        else:
            c = getLn(s[T_CLASS][classes[idx]][COVARIANCE][b]) - getLn(s[T_CLASS][decKey][COVARIANCE][b]) + \
                mahalanobisDistance(x, s[T_CLASS][classes[idx]][MEAN][b], s[T_CLASS][classes[idx]][COVARIANCE][b]) - \
                mahalanobisDistance(x, s[T_CLASS][decKey][MEAN][b], s[T_CLASS][decKey][COVARIANCE][b])
        if c < 0:
            decKey = classes[idx]
    return decKey

def getLn(covarianceMatrix):
    if np.linalg.cond(covarianceMatrix) < 1/sys.float_info.epsilon:
        return np.log(np.linalg.det(covarianceMatrix))
    else:
        d = np.linalg.det(covarianceMatrix)
        if d == 0: return 0 # if d <= 0: return 0
        return np.log(d) if d != 0 else d

def mahalanobisDistance(x, m, c):
    if np.linalg.det(c) == 0.0:
        l = np.linalg.pinv(c)
    else:
        l = np.linalg.inv(c)
    return reduce(np.dot, [np.array(x-m).T, l, x-m])

if __name__ == "__main__":
   #main(WINE_SETTING, OPTIMAL_BAYES)
   #main(IRIS_SETTING, OPTIMAL_BAYES)
   main(HEART_SETTING, OPTIMAL_BAYES)
   #main(HEART_SETTING, NAIVE_BAYES)
   #main(IRIS_SETTING, NAIVE_BAYES)
   #main(WINE_SETTING, NAIVE_BAYES)

# Not needed
# def manualCov(vectorList, mean):
#     c = 0
#     for i in vectorList:
#         x1 = (i - mean)
#         x2 = np.array([i-mean]).T
#         m1 = np.multiply(x1,x2)
#         c += m1
#     return (1/(len(vectorList)-1)) * c
# def getCovarianceMatrix(sampleClassDict, strat):
#     classCov = dict()
#     if strat == LINEAR:
#         return 0
#     else:
#         for key in sampleClassDict.keys():
#             classCov[key] = np.cov(np.array(sampleClassDict[key]).T)
#             if strat == NAIVE_BAYES:
#                 classCov[key] = np.diag(np.diag(classCov[key]))
#     return classCov
# def getMeanByClasses(sampleClassDict):
#     classMeans = dict()
#     for key in sampleClassDict.keys():
#         classMeans[key] = np.mean(sampleClassDict[key], axis=0)
#     return classMeans