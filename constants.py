# Enums
LINEAR, OPTIMAL_BAYES, NAIVE_BAYES, K_FOLD, LOO, DESC = "LINEAR", "OPTIMAL BAYES", "NAIVE BAYES", "10-Fold", "Leave One Out", "Decision Tree"
# Useful settings keys
COVARIANCE, MEAN, CLASS_INDEX, TITLE, LINEAR_COV, TEST_STRATEGY, R_CLASS\
    = "train_classes_cov", "train_classes_mean","classIndex","title","linear_covariance", "test_strategy", "10_fold_raw_class"
TEN_FOLD, T_CLASS, TEST_BATCH, TRAIN, TEST = "10_fold_batches", "10_fold_classes", "test_data", 0, 1


IRIS_SETTING = {"usecols": (0,1,2,3,4), "classIndex":4,"bins":7,
               "data": "./data/iris.csv","numRecords":150,"discreetMethod":"round",
               "title":"IRIS", "discreet":4.3}
WINE_SETTING = {"usecols": (0,1,2,3,4,5,6,7,8,9,10,11,12,13),"classIndex":0,
               "data": "./data/wine.csv","numRecords": 178,"bins":4,
               "title":"WINE", "discreet":4.3, "discreetMethod":"bins"}
HEART_SETTING = {"usecols": (0,2,3,4,6,7,9,10,11,12,13),"classIndex":10,
               "data": "./data/heartDisease.csv","numRecords":297,"bins":6,
               "title":"HEARTDISEASE", "discreet":4.3,"discreetMethod":"bins"}
