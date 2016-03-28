from __future__ import division
import numpy as np
import sklearn.cross_validation as skc
import decisiontree as d
import bayes as b
from constants import *


# Ignore those annoying errors
# np.seterr(invalid='ignore',divide='ignore')



def main(s, strat, testMethod, data=None):
    csvData = data if data is not None else loadCSV(s["data"], s)
    s[TEST_STRATEGY] = testMethod
    if strat == DESC:
        crossValidateSplit(csvData, s, testMethod, True)
        trainDecisionTree(s, strat)
    else:
        crossValidateSplit(csvData, s, testMethod)
        train(s, strat)
        test(s, strat)

def trainDecisionTree(s, strat):
    currentBatch, currentClass, keys, totalCorrectness = 0, 0, s[T_CLASS].keys(), 0
    for key in keys:
        currentClass, classCorrectness = key, 0
        for batchNumber in xrange(len(s[T_CLASS][key][TEST_BATCH])):  # for each K-fold, leave 1 out
            batchDataset, testDataset = [], None
            for key in keys:
                if key == currentClass:
                    batchDataset.append(s[T_CLASS][key][TEST_BATCH][batchNumber][TRAIN])
                    testDataset = s[T_CLASS][key][TEST_BATCH][batchNumber][TEST]
                else:
                    batchDataset.append(s[R_CLASS][key][TEST_BATCH])
            batchDataset = [data for sublist in batchDataset for data in sublist]
            decisionTree = d.DecisionTree(s[CLASS_INDEX])
            root = decisionTree.buildDecisionTree(batchDataset)
           #decisionTree.printtree(root)
            for data in testDataset:
                if key == decisionTree.classify(root,data):
                    print "horray"
                    classCorrectness += 1
                else:
                    print "BOO"
            print "%s class %s for %s accuracy %0.2f%%" % (s[TITLE], str(key), strat,
                                                           (classCorrectness / len(s[T_CLASS][key][TEST_BATCH])) * 100)
    print "%s accuracy using %s for %s dataset is %.2f%% over %d records\n" % (s[TEST_STRATEGY],
                                                                               strat, s[TITLE],
                                                                               100 * (totalCorrectness / len(keys),
                                                                               len(batchDataset) + len(testDataset)))


def loadCSV(path, s):
    return np.genfromtxt(path, dtype=None, usecols=s["usecols"], names=True, delimiter=',')


def processTuple(t, s):
    if t[s[CLASS_INDEX]].dtype.type is np.string_:
        l = [t[x] for x in xrange(s['classIndex'] + 1)]
        return np.asarray(l, dtype=object)
    return np.asarray(np.asarray(t, dtype=object).item(0))


def crossValidateSplit(data, s, strat, leaveAnswer=False):
    classes, s[T_CLASS], s[R_CLASS] = getClasses(data, s, leaveAnswer), dict(), dict()
    for key in classes.keys():
        splitIdx = skc.KFold(len(classes[key]), n_folds=10, shuffle=True) if strat == K_FOLD else skc.LeaveOneOut(
            len(classes[key]))
        s[T_CLASS][key], s[T_CLASS][key][TEST_BATCH] = dict(), []
        for train, test in splitIdx:
            if key not in s[R_CLASS]:
                collectedTuple, s[R_CLASS][key] = ([classes[key][j] for j in train],
                                                   [classes[key][j] for j in test]), dict()
                s[R_CLASS][key][TEST_BATCH] = np.concatenate(collectedTuple)
                s[T_CLASS][key][TEST_BATCH].append(collectedTuple);
            else:
                s[T_CLASS][key][TEST_BATCH].append(([classes[key][j] for j in train],
                                                    [classes[key][j] for j in test]))


def getClasses(tupleArray, s,leaveAnswer):
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


def test(s, strat):
    totalCorrectness, count = 0, 0
    for key in s[T_CLASS].keys():
        classCorrectness = 0
        for batchNumber in xrange(len(s[T_CLASS][key][TEST_BATCH])):
            testSample = s[T_CLASS][key][TEST_BATCH][batchNumber][TEST]
            classCorrect, classTotal = 0, len(testSample)
            for x in testSample:
                classifiedClass = b.classify(s, x, strat, batchNumber, key)
                classCorrect = classCorrect + 1 if key == classifiedClass else classCorrect
                count += 1
            classCorrectness += (classCorrect / classTotal)
        totalCorrectness += classCorrectness / len(s[T_CLASS][key][TEST_BATCH])
        # print "%s class %s for %s accuracy %0.2f%%" % (s[TITLE], str(key), strat,
        #                                                (classCorrectness / len(s[T_CLASS][key][TEST_BATCH])) * 100)
    print "%s accuracy using %s for %s dataset is %.2f%% over %d records\n" % (s[TEST_STRATEGY], strat, s[TITLE],
                                                                               100 * (totalCorrectness / len(
                                                                                   s[T_CLASS].keys())), count)


def train(s, strat):
    for key in s[T_CLASS].keys():
        s[T_CLASS][key][MEAN], s[T_CLASS][key][COVARIANCE], s[R_CLASS][key][MEAN], s[R_CLASS][key][COVARIANCE] = \
            [], [], np.mean(s[R_CLASS][key][TEST_BATCH], axis=0), b.getCovarianceMatrix(s[R_CLASS][key][TEST_BATCH],
                                                                                        strat)
        for i in xrange(len(s[T_CLASS][key][TEST_BATCH])):
            trainSample = s[T_CLASS][key][TEST_BATCH][i][TRAIN]
            s[T_CLASS][key][MEAN].append(np.mean(trainSample, axis=0))
            s[T_CLASS][key][COVARIANCE].append(b.getCovarianceMatrix(trainSample, strat))
    if strat == LINEAR: b.averageCovariance(s)


if __name__ == "__main__":
    wine = loadCSV(WINE_SETTING["data"], WINE_SETTING)
    heart = loadCSV(HEART_SETTING["data"], HEART_SETTING)
    iris = loadCSV(IRIS_SETTING["data"], IRIS_SETTING)
   # main(WINE_SETTING, OPTIMAL_BAYES, K_FOLD, wine)
    # main(WINE_SETTING, OPTIMAL_BAYES, LOO, wine)
    # main(IRIS_SETTING, OPTIMAL_BAYES, K_FOLD, iris)
    # main(IRIS_SETTING, OPTIMAL_BAYES, LOO, iris)
    # main(HEART_SETTING, OPTIMAL_BAYES, K_FOLD, heart)
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
    # main(IRIS_SETTING, DESC, K_FOLD, iris)
    # main(IRIS_SETTING, DESC, LOO, iris)
    # main(HEART_SETTING, DESC, K_FOLD, heart)
    # main(HEART_SETTING, DESC, LOO, heart)
