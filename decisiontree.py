from __future__ import division
from constants import *
import numpy as np
from threading import Thread
import pandas as pd

class DecisionTreeNode(object):
    def __init__(self, col=None, val=None, result=None, t=None, f=None):
        self.columnIndex = col
        self.trueValThreash = val
        self.branchResultDict = result  # None for fringe
        self.trueNodes = t
        self.falseNodes = f

class DecisionTree(object):
    def __init__(self, settings, bins):
        self.classIndex = settings[CLASS_INDEX]
        self.discreet = settings["discreet"]
        self.bins = settings["bins"]
        self.inds = bins

    def getUniqueCounts(self, rows):
        results = {}
        for row in rows:
            # The result is the last column
            r = row[self.classIndex]
            if r not in results: results[r] = 0
            results[r] += 1
        return results

    def entropy(self, rows):
        results = self.getUniqueCounts(rows)
        # Now calculate the entropy
        ent = 0.0
        for  r in results.keys():
            # current probability of class
            p = results[r] / len(rows)
            ent -= p * np.log2(p)
        return ent

    def entropyGain(self, s1, s2):
        b = s1 + s2
        return -(s1 / b) * np.log2(s1 / b) - (1-(s1/b))*(s2 / b) * np.log2(s2 / b)

    def train(self, rows):
        if len(rows) == 0: raise ValueError("Must provide training with non empty dataset")
        # Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} |Sv| / |S| Entropy(Sv)
        df = pd.DataFrame(rows)
        currentGain = self.entropy(rows)
        highestGain, bestAttribute, bestSet, numAttributes = 0.0, None, None, len(rows[0])
        for attribute in xrange(numAttributes):
            if attribute == self.classIndex:
                continue
            uniqueAttributes = set([self.roundingDiscretize(row[attribute]) for row in rows])
            for value in uniqueAttributes:
                # Calculate information gain on this attribute
                s1, s2 = self.partitionDataset(rows, attribute, value)
                if len(s2) == 0:
                    informationGain = 0
                else:
                    p = float(len(s1))/len(s2)
                    informationGain = currentGain - p*self.entropy(s1) - (1-p)*self.entropy(s2)
                if informationGain > highestGain and len(s1) > 0 and len(s2) > 0:
                    highestGain = informationGain
                    bestAttribute = (attribute, value)
                    bestSet = (s1, s2)
        if highestGain > 0:
            trueBranch = self.train(bestSet[0])
            falseBranch = self.train(bestSet[1])
            return DecisionTreeNode(col=bestAttribute[0], val=bestAttribute[1], t=trueBranch, f=falseBranch)
        else:
            return DecisionTreeNode(result=self.getUniqueCounts(rows))

    def partitionDataset(self, rows, attribute, value):
        # gtThread = ThreadWithReturnValue(target=self.gtEqThread, args=(rows, attribute, value))
        # ltThread = ThreadWithReturnValue(target=self.ltThread, args=(rows, attribute, value))
        # gtThread.start()
        # ltThread.start()
        # return gtThread.join(),ltThread.join()
        return [row for row in rows if self.roundingDiscretize(row[attribute]) >= value], \
               [row for row in rows if self.roundingDiscretize(row[attribute]) < value]

    def gtEqThread(self, rows, attribute, value):
        return [row for row in rows if row[attribute] >= value]

    def ltThread(self, rows, attribute, value):
        return [row for row in rows if row[attribute] < value]

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

    def getBinValue(self, row, idx):
        if type(self.inds) == float: return self.roundingDiscretize(row[idx])
        bin, digit = self.inds[idx - 1], 0
        for threash in bin:
            if row[idx] > threash:
                digit += 1
            else: break
        return digit

    def roundingDiscretize(self, int):
        if type(self.inds) == float:
            return round(int*self.inds)/self.inds
        else: return int

    def classify(self, treeRoot, sampleVector):
        node = treeRoot
        while True:
            if node.branchResultDict is not None:
                return node.branchResultDict.keys()[0]
            else:
                #min, max = self.getBinValue(node)
                if self.getBinValue(sampleVector, node.columnIndex) >=  node.trueValThreash:
                    node = node.trueNodes
                else:
                    node = node.falseNodes
