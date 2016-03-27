from constants import  *
import numpy as np
import sys

def getCovarianceMatrix(dataset, strat):
    return np.diag(np.diag(np.cov(np.array(dataset).T))) if strat == NAIVE_BAYES else np.cov(np.array(dataset).T)

def classify(s, x, strat, b, key): # Fix the high coupling this has with (s dict)
    classes = s[T_CLASS].keys()
    if len(classes) < 2: raise ValueError("Cannot classify with only one class")
    decKey = classes[0]
    for idx in range(1, len(classes)):
        bMean, aMean = s[T_CLASS][decKey][MEAN][b] if decKey == key else s[R_CLASS][decKey][MEAN], \
                       s[T_CLASS][classes[idx]][MEAN][b] if classes[idx] == key else s[R_CLASS][classes[idx]][MEAN]
        bCov, aCov = s[T_CLASS][decKey][COVARIANCE][b] if decKey == key else s[R_CLASS][decKey][COVARIANCE], \
                     s[T_CLASS][classes[idx]][COVARIANCE][b] if classes[idx] == key else s[R_CLASS][classes[idx]][COVARIANCE]
        if strat == LINEAR:
            avgCov = (bCov + aCov) / 2
            c = mahalanobisDistance(x,aMean,avgCov) - mahalanobisDistance(x,bMean,avgCov)
        else:
            c = getLn(aCov) - getLn(bCov) + mahalanobisDistance(x, aMean, aCov) - mahalanobisDistance(x, bMean, bCov)
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
    if c is None or x is None or m is None: raise ValueError("Null value passed to mahalanobisDistance")
    if np.linalg.det(c) == 0.0:
        l = np.linalg.pinv(c)
    else:
        l = np.linalg.inv(c)
    return reduce(np.dot, [np.array(x-m).T, l, x-m])

def averageCovariance(s):
    keySet = s[T_CLASS].keys()
    if len(s[T_CLASS][keySet[0]][COVARIANCE]) == 0: raise ValueError("No covariance was calculated.")
    count, s[LINEAR_COV] = 0 , np.zeros(shape=(len(s[T_CLASS][keySet[0]][COVARIANCE][0]),len(s[T_CLASS][keySet[0]][COVARIANCE][0])))
    for key in keySet:
        for i in xrange(len(s[T_CLASS][key][COVARIANCE])):
            s[LINEAR_COV] += s[T_CLASS][key][COVARIANCE][i]
            count += 1
    s[LINEAR_COV] = s[LINEAR_COV]/count