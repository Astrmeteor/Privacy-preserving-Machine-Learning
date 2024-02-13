################################################################################
# LDP-Naive Bayes
################################################################################
# Encoding Methods: Direct (DE), Unary (UE): SUE, OUE, Histogram (HE): SHE, THE
################################################################################
from __future__ import division
import pandas as pd
import numpy as np
from math import ceil
import sys
import os

# 获取当前脚本文件的完整路径
script_path = os.path.abspath(__file__)

# 获取脚本所在目录的路径
script_dir = os.path.dirname(script_path)

# 将当前工作目录更改为脚本所在目录
os.chdir(script_dir)


# Class LDP-Naive Bayes
class ldpnb:
    # dataFrame=None, lblCol=None, X = None, y = None):
    def __init__(self, **kwargs):
        if 'X' in kwargs and 'y' in kwargs:
            dataFrame = pd.DataFrame(np.column_stack((kwargs['y'], kwargs['X'])))
            lblCol = "first"
        elif "dataFrame" in kwargs and "lblCol" in kwargs:
            dataFrame = kwargs['dataFrame']
            lblCol = kwargs['lblCol']
        else:
            sys.exit("INPUT ERROR: No data was provided")
        # Separate the dataFrame into X and y
        if lblCol == "last":
            dfX = dataFrame.iloc[:, :-1] # All except the last column
            dfy = dataFrame.iloc[:, -1]  # Just the last column
        elif lblCol == "first":
            dfX = dataFrame.iloc[:, 1:] # All except the first column
            dfy = dataFrame.iloc[:, 0]  # Just the first column
        else:
            sys.exit("INPUT ERROR - lablel column argument is incorrect")

        # Go over the columns of X, & convert to numerical values
        # Save the dictionary to translate from/to
        self.prepData(dfX, dfy)
        # Initialize several parameters
        self.eps = None
        self.fDict = None
        self.fMap = None
        self.numX = None
        self.labels = None
        self.classes = None
        # Encoding method
        self.method = None
        # inflate tmpX and labels, feature column number
        self.infX = None
        self.infy = None
        self.infFeatColNum = None
        # How many columns were used for features, and for classes
        self.featColNum = None
        self.clColNum = None
        # Initialize the matrix: #samples x (#classes * (#features + 1label) )
        self.fullMat = None
        # start with the test data, training data
        self.tstX = None
        self.tsty = None
        self.trFullMat = None
        # the probabilities p & q
        self.p0 = None
        self.q0 = None
        self.p1 = None
        self.p2 = None
        self.p = None
        self.q = None
        # the perturbed matrix
        self.ldpMat = None
        # Estimate the class labels sum
        self.estClSum = None
        # Adjust the values by subtracting the randomly added values
        self.estAttSum = None
        # Initialize probability vectors
        self.attProb = None
        self.clProb = None

    # prepData: converts data to numerical, and store the mapping
    # Output: fDict {feat1 Idx: ['catVal1', 'catVal2'], feat2 Idx: ['catVal1', 'catVal2']}
    # Output: fMap - no. of categorical values per feature - [2, 2]
    # Output: X - feature data converted to numerical data (for Numpy Array)
    # Output: labels - samples' class labels (y) converted to numerical values
    # Output: classes - class categories mapping (numerical to textual)
    def prepData(self, dfX, dfy):
        # Handle the features first: dfX matrix
        self.fDict = {}
        self.fMap = list()
        self.numX = np.zeros(dfX.shape)
        for i in range(len(dfX.columns)):
            # Extract one column (feature), and make it categorical
            col = dfX.iloc[:, i].astype('category')
            # convert it to numbers (rather than text categories)
            self.numX[:, i] = np.array(col.cat.codes)
            # retrive/store the translation from numbers to text categories
            self.fDict[i] = col.cat.categories.tolist()
            # Set the features' number
            self.fMap.append(len(self.fDict[i]))
        # Handle the labels (turn them into numerical values)
        dfy.is_copy = False
        y = dfy.astype('category')
        self.labels = np.array(y.cat.codes)
        self.classes = y.cat.categories.tolist()

        # Regardless of whether we will have "DE" or other encoding methods, we
        # will create a unary encoded matrix (will be used for testing)
        # Basically, inflate tmpX and labels to 1s and 0s
        # Initialize inflated X: cols = no. of features x no. of categories per feature
        featNum = sum(self.fMap)
        self.infX = np.zeros((self.numX.shape[0], featNum))
        fColIdx = 0
        for i, fVal in enumerate(self.fMap):
            # Set this feature's columns according to X[:, i]
            self.infX[np.arange(self.numX.shape[0]), fColIdx + self.numX[:, i].astype(int)] = 1
            # Increment the column index (fVal is the feature's number of categories)
            fColIdx += fVal
        # Inflate y: cols = no. of classes
        self.infy = np.zeros((len(self.labels), len(self.classes)))
        self.infy[np.arange(self.labels.shape[0]), self.labels.astype(int)] = 1
        # The next parameter WILL PROBABLY BE USED in testing phase
        self.infFeatColNum = self.infX.shape[1]

    # Encode the data into the proper format
    def encode(self, encoding):
        # Encoding Method
        self.method = encoding
        # If DE, construct the training matrix from self.tmpX
        if self.method == "DE":
            X = self.numX
            y = self.labels
        else:
            X = self.infX
            y = self.infy

        # How many columns were used for features, and for classes
        self.featColNum = X.shape[1]
        self.clColNum = 1 if self.method == "DE" else y.shape[1]

        # Create the encoded data matrix
        # Initialize the matrix: #samples x (#classes * (#features + 1label) )
        self.fullMat = np.zeros((X.shape[0], (self.featColNum * len(self.classes) ) + self.clColNum))
        # Fill the matrix with random values first
        if self.method == "DE":
            for clInd in range(len(self.classes)):
                for attInd, attCount in enumerate(self.fMap):
                    self.fullMat[:, (clInd*self.featColNum)+attInd] = np.random.randint(attCount, size=X.shape[0])
        else:
            fColIdx = 0
            for clInd in range(len(self.classes)):
                for attInd, attCount in enumerate(self.fMap):
                    rndVec = np.random.randint(attCount, size=X.shape[0])
                    self.fullMat[np.arange(X.shape[0]), fColIdx + rndVec.astype(int)] = 1
                    # Increment the column index (fVal is the feature's number of categories)
                    fColIdx += attCount

        # Format the data as: sample-of-class0|sample-of-class1|labels
        for i in range(X.shape[0]):
            # Check the label of this sample, and set the corresponding columns
            dataInd = self.labels[i] * self.featColNum
            # Copy the sample to the appropriate columns
            self.fullMat[i, dataInd:(dataInd+self.featColNum)] = X[i, :]
        # Append the labels (y) vector to the end of the matrix
        clInd = len(self.classes) * self.featColNum
        if self.clColNum == 1:
            self.fullMat[:, clInd] = y
        else:
            self.fullMat[:, clInd:(clInd+self.clColNum)] = y

    # Split into training and testing data (When no randomization is required)
    def trainTestFixedSplit(self, trCount, tstCount):
        # If the passed training and testing counts are wrong, just exit()
        if (trCount + tstCount) != self.infX.shape[0]:
            exit()
        # start with the test data
        self.tstX = self.infX[trCount:, :]
        self.tsty = self.labels[trCount:]
        # then the training data
        self.trFullMat = self.fullMat[:trCount, :]

    # Split into training and testing data
    def trainTestSplit(self, trPrcnt=0.8, randomState=42, trSize=2000):
        # Count of training data (we can use: numX, infX, fullMat ... it doesn't matter, we just no. of samples)
        trCount = int(trPrcnt * self.numX.shape[0])
        # "def" means we don't need to replicate training data
        if trSize == "def":
            trSize = trCount
        # Randomly permute the indices
        np.random.seed(randomState)
        indices = np.random.permutation(self.numX.shape[0])
        # Split indices into training and testing
        trIdx, tstIdx = indices[:trCount], indices[trCount:]

        # start with the test data
        self.tstX = self.infX[tstIdx, :]
        self.tsty = self.labels[tstIdx] # self.infy[tstIdx, :]
        # then the training data
        # Compute the multiple we need from the original one
        tmpMat = self.fullMat[trIdx, :]
        reqMult = int(ceil(trSize / tmpMat.shape[0]))
        print(f"REQUESTED" + str(reqMult))
        # repeats the data reqMult times
        self.trFullMat = np.repeat(tmpMat, reqMult, axis=0)
        self.trFullMat = self.trFullMat[:trSize, :]

    # Flip Unfair Coin with prob p, and return a matrix of dim M x N
    def flipUnfairCoin(self, p, M, N, tol=0.1):
        # Variable condition is needed to emulate do-while loop in python
        condition = True
        # While loop until acceptable tolerance is achieved
        while condition:
            # Generate a random matrix
            rndMat = np.random.random((M, N)) if N > 1 else np.random.random(M)
            rndMat = (rndMat < p).astype(int)
            # Do we need to repeat ?
            prop = np.sum(rndMat) / (M * N)
            condition = np.abs(p - prop) > tol
        return rndMat

    # Return the probabilities p & q
    def epsToProb(self, d=1, th=1):
        if self.method == "DE":
            # we don't set global p & q cause it's different for each feature
            p = np.exp(self.eps) / (np.exp(self.eps) + d - 1)
            q = 1.0 / (np.exp(self.eps) + d - 1)
            return p, q
        if self.method == "SUE":
            self.p = np.exp(self.eps/2) / (1 + np.exp(self.eps/2))
            self.q = 1.0 / (1 + np.exp(self.eps/2))
        if self.method == "OUE":
            self.p0 = np.exp(self.eps) / (1 + np.exp(self.eps))
            self.q0 = 1.0 / (1 + np.exp(self.eps))
            self.p1 = self.p2 = 0.5
            # For estimator [ estimateSum() called by pertrub() ]
            self.p = self.p1
            self.q = self.q0
        # For SHE, laplace noise is used (so, we don't need epsToProb)
        if self.method == "THE":
            self.p = 1 - (0.5 * np.exp( (self.eps * (th - 1) ) / 2 ))
            self.q = 0.5 * np.exp( (-0.5 * self.eps * th) / 2 )

    def perturb(self, eps, thresh=1):
        np.random.seed()
        # Initialize the perturbed matrix
        self.ldpMat = np.zeros(self.trFullMat.shape)
        # Set the global eps variable
        self.eps = eps
        # Direct Encoding
        if self.method == "DE":
            # First format the list we will use for perturbation loop
            pertList = np.zeros(self.trFullMat.shape[1])
            pertList[:(self.trFullMat.shape[1]-1)] = np.tile(self.fMap, len(self.classes))
            pertList[self.trFullMat.shape[1]-1] = len(self.classes)
            # Each column of trFullMat will have different "d" value
            # and consequently different probability p
            for j, val in enumerate(pertList): # go over the columns (and their value count)
                # Get p & q for the current feature
                p, _ = self.epsToProb(d=val)
                # Generate a random vector according to probability p (recall q = 1-p)
                K = self.flipUnfairCoin(p, self.trFullMat.shape[0], 1)
                # Loop over all samples (concentrating on this feature only)
                for i in range(self.trFullMat.shape[0]):
                    if K[i] == 1:
                        self.ldpMat[i, j] = self.trFullMat[i, j] # report the truth
                    else:
                        # select one of the other d-1 values
                        sel = np.random.randint(val-1)
                        if sel < self.trFullMat[i, j]:
                            self.ldpMat[i, j] = sel
                        else:
                            self.ldpMat[i, j] = sel + 1 # to skip the true value
            # End of DE perturbation
        # Symmetric Unary Encoding (RAPPOR)
        if self.method == "SUE":
            # Probability of telling the truth
            self.epsToProb(eps)
            # Generate random numbers for RR (in a matrix)
            K = self.flipUnfairCoin(self.p, self.trFullMat.shape[0], self.trFullMat.shape[1])
            # K = np.random.binomial(1, p, fullMat.shape)
            # Perturb (Tell the truth when k[i] is 1, lie otherwise)
            self.ldpMat = (self.trFullMat == K).astype(int)
        # Optimized UE: prob of telling the truth is different for 0's and 1's
        if self.method == "OUE":
            # Probability of telling the truth / lying (for 0's and 1's)
            self.epsToProb(eps) # computes p0, q0, p1, q0
            # Generate random numbers for RR (in a matrix)
            # First compute no. of ones (to use for finding perturbation matrix size)
            numOfOnes = len(self.fMap) * len(self.classes) + 1
            # a vector for 1's with probability p1
            K1 = self.flipUnfairCoin(self.p1, self.trFullMat.shape[0] * numOfOnes, 1)
            # and a vector for 0's with probability p0
            K0 = self.flipUnfairCoin(self.p0, self.trFullMat.shape[0] * (self.trFullMat.shape[1] - numOfOnes), 1)
            # Perturb the ones
            self.ldpMat[self.trFullMat == 1] = (np.ones(K1.shape) == K1).astype(int)
            # Perturb the zeros
            self.ldpMat[self.trFullMat == 0] = (np.zeros(K0.shape) == K0).astype(int)
        if self.method == "SHE" or self.method == "THE":
            # Generate random laplacian noise matrix scaled to 2/eps
            K = np.random.laplace(scale=(2.0/eps), size=self.trFullMat.shape)
            # Add the noise matrix to the original matrix
            self.ldpMat = self.trFullMat + K
            # If it's thresholded (THE): set to 1 if above the threshold, or 0 otherwise
            if self.method == "THE":
                self.ldpMat = (self.ldpMat >= thresh).astype(int)
                self.epsToProb(th=thresh)

    # Estimate the sum using equation (1) in Wang's paper
    @staticmethod
    def estimateSum(obsSum, n, p, q, suppressNeg=True):
        estSum = (obsSum - (n * q)) / (p - q)
        if suppressNeg:
            if type(estSum) is np.ndarray:
                estSum[estSum <= 0] = 1 # at least one sample
            elif estSum <= 0:
                estSum = 1 # at least one sample
        return estSum

    # recall: self.featColNum, self.clColNum
    def aggregate(self):
        # (1) AGGREGATE the OBSERVED SUM of attributes and classes
        # Initialize both att. and class sums
        obsAttSum = np.zeros((len(self.classes), sum(self.fMap)))
        obsClSum = np.zeros(len(self.classes))
        # Direct Encoding
        if self.method == "DE":
            # First, compute class counts
            # Find the index of the labels column
            ind = len(self.classes) * self.featColNum
            # Get the counts (occurence of each label) - note labels are returned sorted
            idx, count = np.unique(self.ldpMat[:, ind], return_counts=True)
            obsClSum[idx.astype(int)] = count
            # Second, compute att counts - Go over all classes
            for clInd in range(len(self.classes)):
                # Starting col. index for current class
                dtCol = clInd * self.featColNum
                attSumIdx = 0
                for attInd, attCount in enumerate(self.fMap):
                    # return values and their counts in this feature's column
                    idx, count = np.unique(self.ldpMat[:, dtCol+attInd], return_counts=True)
                    # Set the appropriate columns in obsAttSum
                    # clCnt is the starting index of current feature in attSum vector
                    # idx is the list of feature numerical categories corresponding to counts
                    # examples: idx = [0, 2, 3]->[2, 3, 1] Result in [2, 0, 3, 1] (notice index 1)
                    obsAttSum[clInd, attSumIdx+idx.astype(int)] = count
                    # Move to the next attribute starting index in attSum vector
                    attSumIdx += attCount
        # Other methods share the same matrix structure
        else:
            # Methods: "SUE", "OUE", "SHE", "THE"
            # First, compute class counts
            # Find the index of the labels column
            ind = len(self.classes) * self.featColNum
            # Get the counts of each class
            obsClSum = np.sum(self.ldpMat[:, ind:], axis=0)
            # Second, compute att counts - Go over all classes
            for clInd in range(len(self.classes)):
                # Starting col. index for current class
                dtCol = clInd * self.featColNum
                # Get the count of each feature (belonging to class clInd)
                obsAttSum[clInd, :] = np.sum(self.ldpMat[:, dtCol:(dtCol+self.featColNum)], axis=0)

        # # Adjust the values by subtracting the randomly added values
        # obsAttSum = self.adjustAttValues(obsClSum, obsAttSum, suppressNeg=False)

        # (2) ADJUST the OBSERVED SUMS using p & q to obtain the ESTIMATED SUMS
        self.estAttSum = np.zeros(obsAttSum.shape)
        self.estClSum = np.zeros(obsClSum.shape)
        # If it's SHE, summation by itself is enough
        if self.method == "SHE":
            self.estAttSum = obsAttSum
            self.estClSum = obsClSum
        # Other than SHE, we need to adjust the observed sum ("DE", "SUE", "OUE", "THE")
        # The same estimator is used, but using different p & q
        if self.method in ["SUE", "OUE", "THE"]:
            self.estAttSum = self.estimateSum(obsAttSum, self.ldpMat.shape[0], self.p, self.q)
            self.estClSum = self.estimateSum(obsClSum, self.ldpMat.shape[0], self.p, self.q)
        # For "DE", each feature was randomized differently according to its domain
        # So, we need to get the p & q for each feature
        if self.method == "DE":
            # First, compute the ESTIMATED class counts
            # Get p & q that were used for perturbing the class labels
            p, q = self.epsToProb(d=len(self.classes))
            # Estimate the class labels sum
            self.estClSum = self.estimateSum(obsClSum, self.ldpMat.shape[0], p, q)
            # Second, compute the ESTIMATED attribute counts
            attSumIdx = 0
            for attCount in self.fMap:
                # Get p & q that were used for perturbing this feature
                p, q = self.epsToProb(d=attCount)
                # Set the appropriate columns in estAttSum
                # clCnt is the starting index of current feature in attSum vector
                self.estAttSum[:, attSumIdx:(attSumIdx+attCount)] = \
                    self.estimateSum(obsAttSum[:, attSumIdx:(attSumIdx+attCount)],
                                     self.ldpMat.shape[0], p, q)
                # Move to the next attribute starting index in attSum vector
                attSumIdx += attCount

        # Adjust the values by subtracting the randomly added values
        self.estAttSum = self.adjustAttValues(self.estClSum, self.estAttSum)

    # Adjust the values by subtracting the randomly added values
    def adjustAttValues(self, clSum, attSum, suppressNeg=True):
        for clInd in range(len(self.classes)):
            fColIdx = 0
            rndCount = self.trFullMat.shape[0] - clSum[clInd]
            for attCount in self.fMap:
                attSum[clInd, fColIdx:(fColIdx+attCount)] -= round(rndCount / attCount)
                # Increment the column index (fVal is the feature's number of categories)
                fColIdx += attCount
        if suppressNeg:
            if type(attSum) is np.ndarray:
                # at least one sample
                attSum[attSum <= 0] = 1
            elif attSum <= 0:
                # at least one sample
                attSum = 1
        return attSum

    # Find the probability of each attribute, and class
    def train(self):
        # Initialize probability vectors
        self.attProb = np.zeros(self.estAttSum.shape)
        self.clProb = np.zeros(self.estClSum.shape)
        # First: Classes' probabilities
        self.clProb = self.estClSum / np.sum(self.estClSum)
        # Second: Attribute probabilities
        for clInd in range(len(self.classes)):
            attProbIdx = 0
            for attCount in self.fMap:
                # attSumIdx is the starting index of current feature in the vector
                self.attProb[clInd, attProbIdx:(attProbIdx+attCount)] = \
                    self.estAttSum[clInd, attProbIdx:(attProbIdx+attCount)] / \
                        np.sum(self.estAttSum[clInd, attProbIdx:(attProbIdx+attCount)])
                # Move to the next attribute starting index in attSum vector
                attProbIdx += attCount

    # Testing NB
    def test(self):
        resClass = [None] * self.tstX.shape[0] # Create Empty list of length shape[0]
        for i in range(self.tstX.shape[0]):
            resProb = np.zeros(len(self.clProb))
            for idx in range(len(self.clProb)):
                resProb[idx] = self.clProb[idx] * np.prod(self.attProb[idx, self.tstX[i] == 1])
            resClass[i] = np.argmax(resProb)
        return np.array(resClass)

    def testAcc(self):
        predy = self.test()
        # Accuracy
        return 100 * (predy == self.tsty).sum() / len(predy)


if __name__ == "__main__":
    # Experiment
    # Encoding Methods: 0 DE, 1 SUE, 2 OUE, 3 SHE, 4 THE
    enc = ["DE", "SUE", "OUE", "SHE", "THE"]

    # Pick a dataset
    dtID = 0
    dNames = ['sample', 'car', 'connect', 'mushroom', 'chess']
    file_path = "datasets"
    fileNames = ['example-train.csv', 'car.data.txt', 'connect-4\\connect-4.data',
                 'mushroom\\agaricus-lepiota.data.csv', 'Chess\\kr-vs-kp.data.txt']
    fileNames = [os.path.join(file_path, dataset_name) for dataset_name in fileNames]
    print(fileNames)

    # The location of labels
    lblCols = ["last", "last", "last", "first", "last"]
    # Test probability
    tstProp = [0.05, 0.06, 0.06, 0.06, 0.06]
    # Random seed
    rndSeed = [8204, 4397, 4219, 8, 5705]

    # Read Data from File
    # filename = os.path.join(os.path.dirname(__file__), fileNames[dtID])
    filename = fileNames[dtID]
    # header = 0
    dataFrame = pd.read_csv(filename, sep='\s*,\s*', header=None, skiprows=1, encoding='ascii', engine='python')

    # initialize the class nb: DataFrame, y-col, and encoding type
    method = "sue"
    nb = ldpnb(dataFrame=dataFrame, lblCol=lblCols[dtID])

    # choose encoding method
    nb.encode(enc[1])

    # training set
    # trPrcnt parameter determines what fraction of the entire dataset will be used as the training set
    # randomState parameter ensures the reproducibility of the dataset split
    # trSize parameter offers an option to directly specify the size of the training set
    nb.trainTestSplit(trPrcnt=0.65, randomState=42, trSize="def")
    # perturbation
    epsilon = 1
    nb.perturb(epsilon)
    # aggregation
    nb.aggregate()
    # train
    nb.train()

    print(f"Accuracy = {nb.testAcc():.2f}%")
