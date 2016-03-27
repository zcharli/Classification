from constants import *

class DecisionTreeNode(object):
    def __init__(self, col, val=None, result=None, t=None,f=None):
        self.columnIndex = col
        self.trueValThreash = val
        self.branchResultDict = result # None for fringe
        self.trueNodes = t
        self.falseNodes = f



def entropy(positiveNegativeSet):
    s1, s2 = len(positiveNegativeSet[0]), len(positiveNegativeSet[1])
    return -(s1 / (s1 + s2)) * np.log2(s1 / (s1 + s2)) - (s2 / (s1 + s2)) * np.log2(s2 / (s1 + s2))

def informationGain(infoSet, attribute):
    ent = entropy(infoSet)
    possibleValues = attribute  # get all possible values
    # categorize each data set based on this attribute
    expectedEntropyValue = 0
    # for all possible values of a:
    #       expectedEntropyValue += (size of Sv set)/(size of s)*entropy(Sv)
    return ent - expectedEntropyValue

def divideSet(rows,column,value):
    return ([row for row in rows if row[column] >= value],
            [row for row in rows if row[column] < value])

