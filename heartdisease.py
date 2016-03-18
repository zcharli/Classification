import main as m

START = 0
CLASS_TYPE_IDX = 11

def trainHeartDisease(split):
    trainSample = split[0]
    testSample = split[1]
    trainClassesDict = m.getClasses(trainSample[0], CLASS_TYPE_IDX)
    trainClassMeansDict = m.getMeanByClasses(trainClassesDict)
    trainCovarianceMatrix = m.getCovarianceMatrix(trainSample[0])
    print trainCovarianceMatrix
