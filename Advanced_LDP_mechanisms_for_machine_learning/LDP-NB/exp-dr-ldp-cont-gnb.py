# LDP with continuous features
from __future__ import division
import numpy as np
import ldp_gaussian_naive_bayes as nb # The github one
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file
import os
import helper as h
import importlib

importlib.reload(h)
importlib.reload(nb)

np.set_printoptions(precision=2)

################################################################################
######### FUNCTIONS

def readDataset(dataset, perc=0.8):
    trainDataFile = os.path.join(os.path.dirname(__file__),
                                 'cont-datasets\\' + dataset)
    data = load_svmlight_file(trainDataFile)
    X, y = data[0], data[1]
    trNum = int(X.shape[0] * perc)
    trainX = X[:trNum, :]
    testX = X[trNum:, :]
    trainy = y[:trNum]
    testy = y[trNum:]
    return trainX.toarray(), trainy, testX.toarray(), testy

if __name__ == "__main__":
    ############################################################
    # Experiment
    # Datasets
    dtID = 0
    drID = 0

    ldpID = 1

    # Vectors
    dtNames = ["australian_scale", "breast-cancer_scale", "diabetes_scale",
               "german.numer_scale"]
    ldpMethods = ["basicOne", "basicAll", "alg2"]
    drMethods = ['None', 'PCA', 'DCA']

    ###############################################################
    # Loop to get results over different epsilons & dataset sizes #
    reps = 100
    # Set epsilon and sizes to loop over
    # epsilons = [0.1, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]
    epsilons = np.hstack((np.arange(0.05, 2, 0.2), np.arange(2, 11, 1)))

    #################### read dataset #####################
    trainX, trainy, testX, testy = readDataset(dtNames[dtID])

    # Dimensionality Reduction / OR NOT
    if drID == 0:
        dimList = [trainX.shape[1]]
    elif drID == 1:
        # PCA
        dimList = range(1, trainX.shape[1])
    elif drID == 2:
        # DCA
        dimList = [1] # For these datasets, it's one
    else:
        exit()

    # Results Matrix: no. of epsilons x 2 columns
    resMat = np.zeros((len(epsilons)+1, 1+len(dimList)))
    # Set first row to the sizes (in case I forget)
    resMat[0, 0] = trainX.shape[0]
    resMat[0, 1:] = dimList
    # Loop over Epsilons
    for eidx, eps in enumerate(epsilons):
        # Set the first column to this epsilon
        resMat[eidx+1, 0] = eps

    #### Loop over the dimensions
    for dind, dimLen in enumerate(dimList):
        # Dimensionality Reduction / OR NOT
        if drID == 0:
            projMat = np.eye(dimLen)
            redDim = dimLen
        else:
            # PCA
            projMat, redDim = h.getProjMat(trainX, trainy, type=drMethods[drID],
                                           dimLen=dimLen, subtype='3')
    
        # Project the data
        pTrainX = np.dot(trainX, projMat[:, 0:redDim])
        pTestX = np.dot(testX, projMat[:, 0:redDim])
        # Scale the values down to -1 to 1 range (for LDP)
        pMaxScaler = MaxAbsScaler().fit(pTrainX)
        # Scale the values down to -1 to 1 range (for LDP)
        spTrainX = pMaxScaler.transform(pTrainX)
        spTestX = pMaxScaler.transform(pTestX)
        # Initialize the GNB object
        pGnb = nb.MyGaussianNB(len(np.unique(trainy)), redDim)
    
    # Loop over all epsilons
    for eidx, eps in enumerate(epsilons):
        # Set initial accuracy
        finalAcc = 0
        print("------------")
        print("Epsilon:", eps)
        # 10 trials and get the average
        for i in range(reps):
            # Perturb and Fit
            pGnb.fitWithLDP(spTrainX.copy(), trainy, eps, method=ldpMethods[ldpID])
            # Perform testing and find the accuracy
            acc = accuracy_score(testy, pGnb.predict(spTestX.copy()))
            # To compute the average accuracy of "reps" repititions
            finalAcc += acc
            # print "Pred Label:", classes[predy]
            # print "True Label:", classes[testy]
            # print "Accuracy is:", acc, "%"
        finalAcc = finalAcc / reps
        resMat[eidx+1, 1+dind] = finalAcc
        print(f"Final Accuracy is: {finalAcc}%")
    
    ################################################
    # Finished Processing - Store results matrix   #
    resFileName = os.path.join(os.path.dirname(__file__),
                               'results\\'+dtNames[dtID] + '_drMethod-'+drMethods[drID] +
                               '_ldpMethod-'+ldpMethods[ldpID]+'.csv')
    np.savetxt(resFileName, resMat, delimiter=',', fmt='%.2f')
