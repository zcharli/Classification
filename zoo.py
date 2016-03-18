import numpy as np
import main as m
START = 1
CLASS_TYPE_IDX = 17

def trainZoo(split):
    trainSample = split[0] # Temporary as this is the first split of 10 fold
    testSample = split[1]
    trainClasses = m.getClasses(trainSample[0], CLASS_TYPE_IDX)
    trainClassMeans = m.getMeanByClasses(trainClasses)