import numpy as np
import main as m
START = 1
CLASS_TYPE_IDX = 16

def trainZoo(split):
    trainSample = split[0] # Temporary as this is the first split of 10 fold
    testSample = split[1]
    trainClassesDict = m.getClasses(trainSample[0], CLASS_TYPE_IDX)
    trainClassMeansDict = m.getMeanByClasses(trainClassesDict)
    trainCovarianceMatrix = m.getCovarianceMatrix(trainSample[0])
    print trainCovarianceMatrix
