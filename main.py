from __future__ import division
import numpy as np
import sklearn.cross_validation as skc
import decisiontree as d
import bayes as b
from constants import *
from threadwithreturn import ThreadWithReturnValue
import time


# Ignore those annoying errors
# np.seterr(invalid='ignore',divide='ignore')


def main(s, testMethod, strat, data=None):
    csvData = data if data is not None else loadCSV(s["data"], s)
    s[TEST_STRATEGY] = testMethod
    crossValidateSplit(csvData, s, strat, testMethod)

def loadCSV(path, s):
    return np.genfromtxt(path, dtype=None, usecols=s["usecols"], names=True, delimiter=',')


def processTuple(t, s):
    if t[s[CLASS_INDEX]].dtype.type is np.string_:
        l = [t[x] for x in xrange(s['classIndex'] + 1)]
        return np.asarray(l, dtype=object)
    return np.asarray(np.asarray(t, dtype=object).item(0))

def getClasses(tupleArray, s, leaveAnswer):
    classes, i = dict(), s[CLASS_INDEX]
    if len(tupleArray) == 0: raise ValueError("No classes to get.")
    for sample in tupleArray:
        sampleClass, sampleFeatures = sample[i], processTuple(sample, s)
        if sampleClass in classes:
            array = np.delete(sampleFeatures, i) if leaveAnswer == False else sampleFeatures
            classes[sampleClass].append(array)
        else:
            array = np.delete(sampleFeatures, i) if leaveAnswer == False else sampleFeatures
            classes[sampleClass] = [array]
    return classes

def crossValidateSplit(data, settings, splitStrat, testMethod):
    leaveAnswer = True if testMethod == DESC else False
    settings.update({TEST_STRATEGY:testMethod, "splitStrat":splitStrat})
    classes = getClasses(data, settings, leaveAnswer)
    kFoldDict = dict()
    for key in classes.keys():
        splitIdx = skc.KFold(len(classes[key]), n_folds=10, shuffle=True) if splitStrat == K_FOLD else skc.LeaveOneOut(
            len(classes[key]))
        kFoldDict[key] = []
        for training, test in splitIdx:
            collectedTuple = ([classes[key][j] for j in training],
                              [classes[key][j] for j in test])
            kFoldDict[key].append(collectedTuple)
    maxBatchNumber = getMaxBatchNumber(kFoldDict)
    totalCorrect = 0
    for batchNumber in xrange(maxBatchNumber):
        testPackage = dict()
        for key in kFoldDict.keys():
            testPackage[key] = dict()
            if batchNumber >= len(kFoldDict[key]):
                # pick random
                testPackage[key][TRAIN] = kFoldDict[key][np.random.randint(len(kFoldDict[key]))][TRAIN]
                testPackage[key][TEST] = kFoldDict[key][np.random.randint(len(kFoldDict[key]))][TEST]
            else:
                testPackage[key][TRAIN] = kFoldDict[key][batchNumber][TRAIN]
                testPackage[key][TEST] = kFoldDict[key][batchNumber][TEST]

        totalCorrect += trainDecisionTree(testPackage, settings) if testMethod == DESC else train(testPackage, settings)
    print "%s accuracy using %s for %s dataset is %.2f%% over %d records\n" % \
          (settings[TEST_STRATEGY], settings["splitStrat"], settings[TITLE], 100*(totalCorrect / maxBatchNumber), \
           settings["numRecords"])

def trainDecisionTree(pkgDict, settings=None):
    totalCorrectness, trainBatch, count = 0, [], 0
    for key in pkgDict.keys():
        trainBatch.append(pkgDict[key][TRAIN])
    trainBatch = [data for sublist in trainBatch for data in sublist]
    discretizedDataset, bins = discretize(trainBatch, settings)
    decisionTree = d.DecisionTree(settings, bins)
    root = decisionTree.train(discretizedDataset)
    # decisionTree.printtree(root)
    for key in pkgDict.keys():
        for sample in pkgDict[key][TEST]:
            if key == decisionTree.classify(root,sample):
                totalCorrectness += 1
            count += 1
    return totalCorrectness / count

def discretize(dataset, settings):
    if settings["discreetMethod"] == "round": return dataset, settings['discreet']
    index, discreetDataset, bins = 0, [], []
    for row in np.array(dataset).T:
        if index == settings[CLASS_INDEX]:
            index += 1
            discreetDataset.append(row)
            continue
        if settings[TITLE] == "IRIS":
            row = np.asarray(row, dtype=float)
        bins.append(np.linspace(row.min(), row.max(), settings["bins"]))
        discreetDataset.append(np.digitize(row, bins[-1]))
        index += 1
    return np.array(discreetDataset).T, bins


def train(pkgDict, settings=None):
    test, classCorrect, count = [], 0, 0
    for key in pkgDict.keys():
        matrix = [np.asarray(item) for item in pkgDict[key][TRAIN]]
        pkgDict[key][COVARIANCE] = b.getCovarianceMatrix(matrix, settings[TEST_STRATEGY])
        pkgDict[key][MEAN] = np.mean(pkgDict[key][TRAIN], axis=0)
        test.append(pkgDict[key][TEST])
    for key in pkgDict.keys():
        for sample in pkgDict[key][TEST]:
            classifiedClass = b.classify(pkgDict, sample, settings[TEST_STRATEGY])
            classCorrect = classCorrect + 1 if key == classifiedClass else classCorrect
            count += 1
    return classCorrect/count

def getMaxBatchNumber(dic):
    maxBatch = 0;
    for key in dic.keys():
        if len(dic[key]) > maxBatch:
            maxBatch = len(dic[key])
    return maxBatch

if __name__ == "__main__":
    wine = loadCSV(WINE_SETTING["data"], WINE_SETTING)
    heart = loadCSV(HEART_SETTING["data"], HEART_SETTING)
    iris = loadCSV(IRIS_SETTING["data"], IRIS_SETTING)
    #main(WINE_SETTING, OPTIMAL_BAYES, K_FOLD, wine)
    #main(WINE_SETTING, OPTIMAL_BAYES, LOO, wine)
    #main(IRIS_SETTING, OPTIMAL_BAYES, K_FOLD, iris)
    #main(IRIS_SETTING, OPTIMAL_BAYES, LOO, iris)
    #main(HEART_SETTING, OPTIMAL_BAYES, K_FOLD, heart)
    # main(HEART_SETTING, OPTIMAL_BAYES, LOO, heart)
    # main(WINE_SETTING, NAIVE_BAYES, K_FOLD, wine)
    # main(WINE_SETTING, NAIVE_BAYES, LOO, wine)
    # main(IRIS_SETTING, NAIVE_BAYES, K_FOLD, iris)
    # main(IRIS_SETTING, NAIVE_BAYES, LOO, iris)
    # main(HEART_SETTING, NAIVE_BAYES, K_FOLD, heart)
    # main(HEART_SETTING, NAIVE_BAYES, LOO, heart)
    # main(WINE_SETTING, LINEAR, K_FOLD, wine)
    # main(WINE_SETTING, LINEAR, LOO, wine)
    # main(IRIS_SETTING, LINEAR, K_FOLD, iris)
    # main(IRIS_SETTING, LINEAR, LOO, iris)
    # main(HEART_SETTING, LINEAR, K_FOLD, heart)
    # main(HEART_SETTING, LINEAR, LOO, heart)
    main(WINE_SETTING, DESC, K_FOLD, wine)
    # main(WINE_SETTING, DESC, LOO, wine)
    main(IRIS_SETTING, DESC, K_FOLD, iris)
    #main(IRIS_SETTING, DESC, LOO, iris)
    #main(HEART_SETTING, DESC, K_FOLD, heart)
    # main(HEART_SETTING, DESC, LOO, heart)



# def decisionTreeThread(s, batchNumber, keys, currentClass, key):
#     batchDataset, testDataset, classCorrect = [], None, 0
#     for eachKey in keys:
#         if eachKey == currentClass:
#             batchDataset.append(s[T_CLASS][eachKey][TEST_BATCH][batchNumber][TRAIN])
#             testDataset = s[T_CLASS][eachKey][TEST_BATCH][batchNumber][TEST]
#         else:
#             batchDataset.append(s[R_CLASS][eachKey][TEST_BATCH])
#     batchDataset = [data for sublist in batchDataset for data in sublist]
#     decisionTree = d.DecisionTree(s[CLASS_INDEX])
#     root = decisionTree.train(batchDataset)
#     decisionTree.printtree(root)
#     for data in testDataset:
#         if key == decisionTree.classify(root, [int(dis) for dis in data]):
#             classCorrect += 1
#     return classCorrect, batchDataset, testDataset
