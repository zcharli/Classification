import main as m
START = 2
CLASS_TYPE_IDX = 9

def trainHeartDisease(split):
    trainSample = split[0]
    testSample = split[1]
    trainClasses = m.getClasses(trainSample, CLASS_TYPE_IDX)

