import numpy as np
import main as m
START = 1
CLASS_TYPE_IDX = 16

def trainZoo(zooSettingsDict):
    for batches in zooSettingsDict["10_fold_batches"]:
        trainSample = batches[0]
        testSample = batches[1]
        zooSettingsDict["train_classes_dict"] = m.getClasses(trainSample)
        zooSettingsDict["train_classes_mean_dict"] = m.getMeanByClasses(zooSettingsDict["train_classes_dict"])
        zooSettingsDict["train_classes_cov_dict"] = m.getCovarianceMatrix(zooSettingsDict["train_classes_dict"])
        print zooSettingsDict["train_classes_cov_dict"]
        m.classify(zooSettingsDict)