from __future__ import division
from constants import *
import numpy as np


class DecisionTreeNode(object):
    def __init__(self, col=None, val=None, result=None, t=None, f=None):
        self.columnIndex = col
        self.trueValThreash = val
        self.branchResultDict = result  # None for fringe
        self.trueNodes = t
        self.falseNodes = f

class DecisionTree(object):
    def __init__(self, classIndex):
        self.classIndex = classIndex

    def train(self, dataset):
        return self.buildDecisionTree(dataset)

    # Create counts of possible results (last column of each row is the result)
    def uniqueCounts(self, rows):
        results = {}
        for row in rows:
            # The result is the last column
            r = row[self.classIndex]
            if r not in results: results[r] = 0
            results[r] += 1
        return results

    def entropy(self, rows):
        results = self.uniqueCounts(rows)
        # Now calculate the entropy
        ent = 0.0
        for r in results.keys():
            # current probability of class
            p = results[r] / len(rows)
            ent -= p * np.log2(p)
        return ent

    def entropyGain(self, s1, s2):
        b = s1 + s2
        return -(s1 / b) * np.log2(s1 / b) - (1-(s1/b))*(s2 / b) * np.log2(s2 / b)

    def buildDecisionTree(self, rows):
        if len(rows) == 0: return None
        # Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} |Sv| / |S| Entropy(Sv)
        currentGain = self.entropy(rows)
        highestGain, bestAttribute, bestSet, numAttributes = 0.0, None, None, len(rows[0])
        for attribute in xrange(numAttributes):
            if attribute == self.classIndex:
                continue
            uniqueAttributes = set([row[attribute] for row in rows])
            for value in uniqueAttributes:
                # Calculate information gain on this attribute
                s1, s2 = self.divideSet(rows, attribute, value)
                if len(s2) == 0:
                    gain = 0
                else:
                    p = float(len(s1))/len(s2)
                    gain = currentGain - p*self.entropy(s1) - (1-p)*self.entropy(s2)
                if gain > highestGain and len(s1) > 0 and len(s2) > 0:
                    highestGain = gain
                    bestAttribute = (attribute, value)
                    bestSet = (s1, s2)
        if highestGain > 0:
            trueBranch = self.buildDecisionTree(bestSet[0])
            falseBranch = self.buildDecisionTree(bestSet[1])
            return DecisionTreeNode(col=bestAttribute[0], val=bestAttribute[1], t=trueBranch, f=falseBranch)
        else:
            return DecisionTreeNode(result=self.uniqueCounts(rows))

    def divideSet(self, rows, attribute, value):
        return ([row for row in rows if row[attribute] >= value],
                [row for row in rows if row[attribute] < value])

    def printtree(self, tree, indent=''):
        # Is this a leaf node?
        if tree.branchResultDict is not None:
            print str(tree.branchResultDict)
        else:
            # Print the criteria
            print 'Column ' + str(tree.columnIndex) + ' : ' + str(tree.trueValThreash) + '? '
            # Print the branches
            print indent + 'True->',
            self.printtree(tree.trueNodes, indent + '  ')
            print indent + 'False->',
            self.printtree(tree.falseNodes, indent + '  ')

    def classify(self, treeRoot, sampleVector):
        node = treeRoot
        while True:
            if node.branchResultDict is not None:
                return node.branchResultDict
            else:
                if sampleVector[node.columnIndex] <= node.trueValThreash:
                    node = node.trueNodes
                else:
                    node = node.falseNodes
