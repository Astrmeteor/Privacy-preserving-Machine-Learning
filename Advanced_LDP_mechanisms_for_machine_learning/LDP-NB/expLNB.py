from __future__ import division
import pandas as pd
import numpy as np
import os
import gldpnb as lnb

reload(lnb)
############################################################
############################ Experiment
# Encoding Methods: 0 DE, 1 SUE, 2 OUE, 3 SHE, 4 THE
enc = ["DE", "SUE", "OUE", "SHE", "THE"]
# Pick Encoding
encID = 4
# Pick a dataset
dtID = 1
dNames = ['sample', 'car', 'connect', 'mushroom', 'chess']
fileNames = ['datasets\\example-trainL.csv', 'datasets\\car.data.txt', \
    'datasets\\connect-4\\connect-4.data', \
    'datasets\\mushroom\\agaricus-lepiota.data.csv', \
    'datasets\\Chess\kr-vs-kp.data.txt']
lblCols = ["last", "last", "last", "first", "last"]
tstProp = [0.05, 0.06, 0.06, 0.06, 0.06]
rndSeed = [1801,  496, 9999,   10,  117]

# Read Data from File
filename = os.path.join(os.path.dirname(__file__), fileNames[dtID])
df = pd.read_csv(filename, sep='\s*,\s*', header=None, skiprows=1, encoding='ascii', engine='python') # header=0

# initialize the class nb: DataFrame, y-col, and encoding type
nb = lnb.ldpnb(dataFrame=df, lblCol=lblCols[dtID])

nb.encode(enc[encID])

nb.trainTestSplit(trPrcnt=(1-tstProp[dtID]), randomState=rndSeed[dtID], trSize="def")

np.set_printoptions(linewidth=120, nanstr='nan', precision=3)

###############################################################
# Loop to get results over different epsilons & dataset sizes #
reps = 100
# Set epsilon and sizes to loop over
# epsilons = [0.1, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]
epsilons = np.hstack((np.arange(0.05, 2, 0.2), np.arange(2, 11, 1)))

# Results Matrix: no. of epsilons x 2 columns
resMat = np.zeros((len(epsilons)+1, 2))
# Set first row to the sizes (in case I forget)
resMat[0, 1:] = nb.trFullMat.shape[0]
# Loop over Epsilons
for eidx, eps in enumerate(epsilons):
    # Set the first column to this epsilon
    resMat[eidx+1, 0] = eps

# Loop over all epsilons
for eidx, eps in enumerate(epsilons):
    # Set initial accuracy
    finalAcc = 0
    print "------------"
    print "Epsilon:", eps
    # 10 trials and get the average
    for i in range(reps):
        # Perturb the data
        nb.perturb(eps)
        # Aggregate the perturbed data
        nb.aggregate()
        # Train (compute the probabilities)
        nb.train()
        # Perform testing and find the accuracy
        acc = nb.testAcc()
        # To compute the average accuracy of "reps" repititions
        finalAcc += acc
        # print "Pred Label:", classes[predy]
        # print "True Label:", classes[testy]
        # print "Accuracy is:", acc, "%"
    finalAcc = finalAcc / reps
    resMat[eidx+1, 1] = finalAcc
    print
    print "Final Accuracy is:", finalAcc, "%"
    print
        

################################################
# Finished Processing - Store results matrix   #
resFileName = os.path.join(os.path.dirname(__file__), \
    'results\\'+dNames[dtID]+'_encMethod_'+enc[encID]+'.csv')
np.savetxt(resFileName, resMat, delimiter=',', fmt='%.2f')