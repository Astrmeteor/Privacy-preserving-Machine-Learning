from __future__ import division
import numpy as np
import scipy.linalg as la
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from sklearn import random_projection
#import arraysetops as oldnp
import math
import sys

def getFeatIndices(system):
    serwadda = np.array([28, 35, 36, 15, 37, 38, 39, 40, 23, 41, 42, 43, 44, \
            45, 46, 47, 48, 49, 50, 51, 5, 6, 7, 8, 4, 9, 12, 26]) - 1 
#    polish = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 25, 26, 28, \
#            31, 32]) - 1 
    unobservable = np.array([5, 6, 52, 53, 26, 4, 25, 54, 47, 55]) - 1 
    xu37 = np.array([5, 6, 52, 56, 57, 58, 59, 60, 61, 62, 7, 8, 63, 64, \
            65, 28, 66, 67, 68, 69, 70, 71, 72, 73, 74, 53, 75, 26, 27, 76, \
            47, 42, 77, 48, 43]) - 1 
    frank = np.array([3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, \
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]) - 1
    # SETS of FEATURES, NOT SYSTEMS 
    allDirFeats = list(np.array([5, 6, 7, 8, 12, 10, 25, 53, 54, 68, 71, \
            74, 75]) - 1)
    allFeats = list(set(serwadda) | set(unobservable) | set(xu37) | set(frank))
    # The "- 1" at the end is because of MATLAB indexing
    if system == 'serwadda':
        return serwadda
    if system == 'serwaddaNoDir':
        return [x for x in serwadda if x not in allDirFeats]
#    if system == 'polish':
#        return polish
    if system == 'unobservable':
        return unobservable
    if system == 'unobservableNoDir':
        return [x for x in unobservable if x not in allDirFeats]
    if system == 'xu37':
        return xu37
    if system == 'xu37NoDir':
        return [x for x in xu37 if x not in allDirFeats]
    if system == 'frank':
        return frank
    if system == 'frankNoDir':
        return [x for x in frank if x not in allDirFeats]
    if system == 'all':
        return allFeats
    if system == 'allNoDir':
        return [x for x in allFeats if x not in allDirFeats]
    return np.array([])
    
def processData(dataMat, sessions, classifiers, system, privTgt='direction'):
    if privTgt == 'gender':
        privCol = 77 # 78 - 1 (because MATLAB starts from 1, Python from 0)
    elif privTgt == 'experience':
        privCol = 78 # 79 - 1 (because MATLAB starts from 1, Python from 0)
    elif privTgt == 'direction': # Default case
        privCol = 10 # 11 - 1 (because MATLAB starts from 1, Python from 0)
    else:
        print "No valid privacy target was given\nTerminating the script"
        sys.exit()
    
    # Get the featureset
    featureset = getFeatIndices(system)
    print featureset
    
    # Select the rows with the specified sessions
    dataMat = dataMat[np.in1d(dataMat[:, 1], sessions), :]
    
    # Select the rows with the specified classifiers
    dataMat = dataMat[np.in1d(dataMat[:, 10], classifiers), :]
    
    # Remove rows with NaN
    dataMat = dataMat[~np.isnan(dataMat).any(axis=1)]
    
    # Only keep the relevant columns - features
    y = dataMat[:, 0]
    X = dataMat[:, featureset]
    
    # Convert features to Z-score - aka standardize
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X) #defaults
    X = scaler.transform(X)
    # We can basically call fit_transform(X) on the training data, and then call 
    # transform(X) on the testing data.
    # We can also store mean_, var_ and scale_ to be used later   
    
    # Was classifier column included? (it's no. 10)
    if np.any(featureset == 10):
        idx = np.where(featureset == 10)[0][0]
        # Remove the classifier column from X
        X = np.c_[X[:, :idx], X[:, idx+1:]]
    
    # Append the privacy target to be the first column
    X = np.c_[dataMat[:, privCol], X]
    
    # Split to training/testing sets
    randomState = np.random.RandomState(0) # WAS 0
    Xtrain, Xtest, ytrain, ytest = \
        train_test_split(X, y, train_size=.8, stratify=y, random_state=randomState)

    # Stack'em together
    trainMat = np.hstack((ytrain[:, np.newaxis], Xtrain))
    testMat  = np.hstack((ytest[:, np.newaxis], Xtest))

    # Return all the results    
    return (trainMat, testMat, scaler)

def prepareForTraining(X, y, cList, posClass=None, totalSize = 4000):
    # Find labels and counts of each class in the training set
    labels, counts = np.unique(y, return_counts=True)
    if posClass != None:
        numPosSamples = counts[labels == posClass][0]
        # If no. of +ve samples is greater than totalSize, we get a problem in 
        # Computing numPerNegClass. Limit +ve samples to 1/3 totalSize, at most
        numPosSamples = min(int(totalSize / 3), numPosSamples)
    else:
        numPosSamples = 0
    # Determine the upper bound on no. of samples per each negative class
    numPerNegClass = \
        int((totalSize - numPosSamples) / (len(labels) - 1))
    
    # Set the no. of samples per class
    # 1- Leave the positive one, and the -ve ones (below the threshold) unchanged
    # 2- Only lower the -ve ones that are above the threshold
    tmpC = counts - numPerNegClass
    counts[tmpC > 0] = numPerNegClass # Cut 
    if posClass != None:
        counts[labels == posClass] = numPosSamples # Return it to original
    
    # Initialize arrays
    Xtrain = np.empty((0, X.shape[1]))
    ytrain = np.empty((0))
    ctrain = np.empty((0))
    Xlo = np.empty((0, X.shape[1]))
    ylo = np.empty((0))
    clo = np.empty((0))
    
    # Assemble the arrays
    for label in labels:
        numPerLabel = counts[labels == label][0]
        allIndices = np.where(y == label)
        trIndices = allIndices[0][:numPerLabel]
        Xtrain = np.append(Xtrain, X[trIndices, :], axis=0)
        ytrain = np.append(ytrain, y[trIndices], axis=0)
        ctrain = np.append(ctrain, cList[trIndices], axis=0)
        # Do we have more samples that we can handle for training (Left overs)
        if len(allIndices[0]) > numPerLabel:
            loIndices = allIndices[0][numPerLabel:]
            Xlo = np.append(Xlo, X[loIndices, :], axis=0)
            ylo = np.append(ylo, y[loIndices], axis=0)
            clo = np.append(clo, cList[loIndices], axis=0)

    return (Xtrain, ytrain, ctrain, Xlo, ylo, clo)

def prepareForTrainingWS(X, y, posClass, cList, totalSize = 4000):
    # Before we do anything, let's put all positive class samples in 
    # Xtrain, ytrain, and remove from X and y
    Xtrain = X[y == posClass]
    ytrain = y[y == posClass]
    # Leave what remains after removing the positive class
    negIndices = np.where(y != posClass)
    X     = X[negIndices]
    y     = y[negIndices]
    cList = cList[negIndices]
    # Update total size
    totalSize = max(totalSize - len(ytrain), 2000)
    
    # Find labels and counts of each class in the training set
    labels, counts = np.unique(y, return_counts=True)
    # Determine the upper bound on no. of samples per each negative class
    numPerNegClass = int(totalSize / len(labels))
    
    # Set the no. of samples per class
    # 1- Leave the positive one, and the -ve ones (below the threshold) unchanged
    # 2- Only lower the -ve ones that are above the threshold
    tmpC = counts - numPerNegClass
    counts[tmpC > 0] = numPerNegClass # Cut off
    
    # Set empty array
    trIndices = np.empty((0))
    
    # Assemble the arrays
    for label in labels:
        numPerLabel = counts[labels == label][0]
        allIndices = np.where(y == label)
        
        # Find what is the count of each cList item for this label
        # Get the different featVals
        Cs, cCounts = np.unique(cList[allIndices], return_counts=True)
        perCSize = (numPerLabel * cCounts / sum(cCounts)).astype(int)
        
        for c in Cs:
            # How many indices do we need?
            perC = perCSize[Cs == c][0]
            # Form an intersection
            cIndices = np.where(cList == c)
            interIndices = np.intersect1d(allIndices, cIndices)
            # Append
            trIndices = np.append(trIndices, \
                interIndices[:min(perC, len(interIndices))], axis=0)

    # Finally, form the final arrays
    trIndices = trIndices.astype(int)
    Xtrain = np.vstack((Xtrain, X[trIndices, :]))
    ytrain = np.hstack((ytrain, y[trIndices]))
    ctrain = cList[trIndices]
    # loIndices is anything left over. Not selected by trIndices
    Xlo = np.delete(X, trIndices, 0)
    ylo = np.delete(y, trIndices, 0)

    return (Xtrain, ytrain, Xlo, ylo, ctrain)

def printStats(X, y, c):
    # Select the rows with the specified classifiers
    dataMat1 = X[c == 1, :]
    dataMat2 = X[c == 2, :]
    dataMat3 = X[c == 3, :]
    dataMat4 = X[c == 4, :]
    y1 = y[c == 1]
    y2 = y[c == 2]
    y3 = y[c == 3]
    y4 = y[c == 4]
    print dataMat1.shape[0], dataMat2.shape[0], dataMat3.shape[0], dataMat4.shape[0]
    
    labels = np.unique(y)
    stats = np.empty([len(labels), 5])
    
    for label in labels:
        stats[int(label-1), 0] = label
        stats[int(label-1), 1] = dataMat1[y1 == label, :].shape[0]
        stats[int(label-1), 2] = dataMat2[y2 == label, :].shape[0]
        stats[int(label-1), 3] = dataMat3[y3 == label, :].shape[0]
        stats[int(label-1), 4] = dataMat4[y4 == label, :].shape[0]
        
    print stats

def rescale(data, paramsFile):
    # Read the file
    params = np.load(paramsFile)
    # Start a scaler class, and set its parameters
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.scale_ = params['scale_']
    scaler.mean_ = params['mean_']
    scaler.var_ = params['var_']
    # Transform the data back to original ranges
    return scaler.inverse_transform(data)
    
#####################################################
# Touchalytics Specific preprocessing and functions #

# Input: coords is formed of [start x, start y, end x, end y] 
# Basically, [5, 6, 7, 8] featurs in MATLAB (-1 for Python, so [4, 5, 6, 7])
def getDirectionFlag(angle):
    # Change to [0, 2pi]
    angle = angle + math.pi
    # Check the direction
    if angle <= (0.25 * math.pi):
        return 4 # Right
    elif angle > (0.25 * math.pi) and angle <= (1.25 * math.pi):
        if angle < (0.75 * math.pi):
            return 1 # UP
        else: 
            return 2 # Left
    else:
        if angle < (1.75 * math.pi):
            return 3 # Down
        else:
            return 4 # Right

def getDirectionAngle(coords):
    # Find angle of direction
    angle = math.atan2(coords[3]-coords[1], coords[2]-coords[0])
    return angle

#####################################################
#### Scatter Matrices, DCA, MDA, RUCA and others ####
def computeShare(X, y, labels):
        # Initialize variables
        numRows, numFeats = X.shape
        
        numLabels = len(labels)
        R = np.zeros((numFeats, numFeats))
        V = np.zeros((numLabels, numFeats))
        N = np.zeros(numLabels)
        # Loop over all samples
        for i in range(numRows):
            # Update R
            R += np.outer(X[i,:], X[i,:])
            # get Index of current class label
            indLabel = np.where(labels==y[i])
            # Update V & N shares
            V[indLabel, :] += X[i, :]
            N[indLabel] += 1
        
        return (R, V, N)

def computeSmatrices(X, y):
    # Set variables
    numRows, numFeats = X.shape
        
    # Initialize R, V and N
    labels = np.unique(y)
    numLabels = len(labels)
    
    # We only have one data owner in this scenario
    (R, V, N) = computeShare(X, y, labels)
        
    # Data User now computes S, Sw and Sb
    # Starting with S (Scatter Matrix)
    sumV = V.sum(axis=0) # Column-wise sum (like MATLAB sum)
    S = R - (1.0/sum(N)) * (np.outer(sumV, sumV))
        
    # Then, Sw (Noise Matrix OR Within-class scatter matrix)
    Sw = R
    for i in range(numLabels):
        Sw -= (1.0/N[i]) * np.outer(V[i,:], V[i,:])
        
    # Finally, Sb (Between-class Scatter Matrix)
    mu = (1.0/sum(N)) * sumV
    Sb = np.zeros((numFeats, numFeats))
    for i in range(numLabels):
        cMu = (1.0/N[i]) * V[i, :]
        Sb += ( N[i] * np.outer(cMu-mu, cMu-mu) )

    return (S, Sw, Sb)

def sortEigVecs(lmda, V):
    # ascending indices
    idx = np.argsort(lmda) 
    # reverse to have descending indices
    ordered_idx = idx[::-1] 
    # get principle components in each column
    V = V[:, ordered_idx]
    # Done
    return (lmda, V)

def computeDCA(X, y, rho=0.001, type=1):
    # Compute S matrices
    (S, Sw, Sb) = computeSmatrices(X, y)
    # Add the ridge value
    ridgeM = rho * np.eye(Sw.shape[0])
    # Which type of DCA is it? Maximizing S or Sb
    # Sw^-1 * S    # Original Thee Code
    if type == 1: 
        (lmda, V) = la.eig(S, Sw + ridgeM)
    # Sw^-1 * Sb   # MDA (Multiclass LDA)
    if type == 2: 
        (lmda, V) = la.eig(Sb, Sw + ridgeM)
    # S^-1 * Sb    # Dr. Kung and Mert papers
    if type == 3:
        (lmda, V) = la.eig(Sb, S + ridgeM)
    # Sort Eigenvalues and vectors
    (lmda, V) = sortEigVecs(lmda.real, V.real)
    return (lmda, V)

def computeJointDCA(X, y, c, rho=0.001, ratio=1.0, type=1):
    # Compute S matrices
    (S, Sw_util, Sb_util) = computeSmatrices(X, y)
    (_, Sw_priv, Sb_priv) = computeSmatrices(X, c)
    # Create the ridge values matrix
    ridgeM = rho * np.eye(Sw_util.shape[0])
    # Which type of DCA is it? Maximizing S or Sb
    
    # Sb_priv^-1 * Sb_util    # MDR (Multiclass Disc. Ratio) 1st Kung
    if type == 1:
        (lmda, V) = la.eig(Sb_util + ridgeM, Sb_priv + ridgeM)
    
    # (S + ratio * Sb_priv)^-1 * Sb_util   # RUCA (Mert Al paper)
    if type == 2:
        (lmda, V) = la.eig(Sb_util, S + (ratio * Sb_priv) + ridgeM)
        
    # Kung's paper (DUCA) # 1   (Equation 37 of IEEE SP Mag. paper)
    # S^-1 * (alpha * Sb_util - Beta * Sb_priv)   
    if type == 3:
        (lmda, V) = la.eig(Sb_util - (ratio * Sb_priv), S + ridgeM)
    
    # Kung's paper (DUCA) # 2   (Last eq. in the IEEE SP Mag. paper)
    # (S - rho * Sb_priv)^-1 * (alpha * Sb_util - Beta * Sb_priv)   
    if type == 4:
        (lmda, V) = la.eig(Sb_util - (ratio * Sb_priv), S + (rho * Sb_priv) + ridgeM)
    
    # (Sw_util + ratio * Sb_priv)^-1 * Sb_util   # WHY NOT?
    if type == 5:
        (lmda, V) = la.eig(Sb_util, Sw_util + (ratio * Sb_priv) + ridgeM)
        
    # (Sw_util + ratio * Sb_priv)^-1 * (Sb_util - ratio * Sw_priv)   # JUPA-2 (By Di)
    if type == 6:
        (lmda, V) = la.eig(Sb_util + (ratio * Sw_priv), Sw_util + (ratio * Sb_priv) + ridgeM)
    
    # Sort Eigenvalues and vectors
    (lmda, V) = sortEigVecs(lmda.real, V.real)
    return (lmda, V)

# Consider this function to be a router 
def getProjMat(X, y, type='NoDimRed', dimLen='def', subtype='1', priv='', rndSeed=0):
    dimLen = str(dimLen)
    # Handle the dimensions
    if dimLen == 'def':
        supRedDim = min(len(np.unique(y)) - 1, X.shape[1])
    elif dimLen == 'full':
        supRedDim = X.shape[1]
    elif dimLen.isdigit() and isinstance(int(dimLen), (int, long)):
        supRedDim = int(dimLen)
    else:
        print "PROBLEM WITH THE INPUT 'DimLen'  \nTerminating This Program"
        exit()
    # Change Sub-Type to integer (proper input)
    if subtype.isdigit():
        subtype = int(subtype)
    else:
        print "PROBLEM WITH THE INPUT 'subtype'  \nTerminating This Program"
        exit()   
    
    # Original data (no dimensionality reduction)
    if type == 'NoDimRed':
        return (np.eye(X.shape[1]), X.shape[1])
        
    # Random Projection (Non-orthogonal vectors)
    if type == 'RND' and subtype == 1:
        print "RANDOM PROJECTION - NON-ORTHOGONAL"
        random_state = check_random_state(rndSeed)
        rndProj = random_projection.GaussianRandomProjection(n_components=supRedDim, \
            random_state=random_state)
        rndProj.fit(X)
        return (rndProj.components_.T, supRedDim)
        
    if type == 'RND' and subtype == 2:
        print "RANDOM PROJECTION - ORTHOGONAL"
        random_state = check_random_state(rndSeed)
        # Generate the random matrix
        rndM = random_state.rand(X.shape[1], supRedDim)
        # Orthogonalize using QR
        rndProj, _ = np.linalg.qr(rndM)
        return (rndProj, supRedDim)

    # PCA
    if type == 'PCA':
        pca = PCA()
        pca.fit(X)
        return (pca.components_.T, supRedDim)
    
    # DCA 
    if type == 'DCA':
        # 4- Perform DCA and project the data
        (lmda, V) = computeDCA(X, y, rho=0.001, type=subtype)
        return (V, supRedDim)
    
    # Joint Utility Privacy (1-MDR, 2-RUCA, 3-DUCA1, 4-DUCA2, 5-WHAT, 6-JUPA2)
    if type == 'JUP':
        # 4- Perform DCA and project the data
        (lmda, V) = computeJointDCA(X, y, priv, rho=0.001, ratio=1.0, type=subtype)
        return (V, supRedDim)

# Recall LDA: 
# eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# eigvec_sc = eig_vecs[:,i].reshape(4,1)
# print eigvec_sc.real

# Further Notes:
# https://dyinglovegrape.wordpress.com/2010/11/30/the-inverse-of-an-orthogonal-matrix-is-its-transpose/
# https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
# http://sebastianraschka.com/Articles/2014_python_lda.html#21-within-class-scatter-matrix-s_w

# Some code

# # TEST Cases
# X = np.array([[0, 7, 8, 1], [1, 1, 2, 2], [2, 7, 8, 3], [3, 3, 4, 1], \
#     [4, 7, 8, 2], [5, 1, 2, 3], [6, 7, 8, 1], [7, 5, 6, 2], [8, 1, 2, 3], \
#     [9, 3, 4, 1], [10, 1, 2, 2], [11, 5, 6, 3], [12, 5, 6, 1], [13, 3, 4, 2], \
#     [14, 7, 8, 3], [15, 7, 8, 1], [16, 1, 2, 2], [17, 5, 6, 3], [18, 7, 8, 1], \
#     [19, 9, 10, 2], [20, 9, 10, 3], [21, 9, 10, 1], [22, 9, 10, 2], \
#     [23, 1, 2, 3], [24, 1, 2, 1]])
# y = np.array([4, 1, 4, 2, 4, 1, 4, 3, 1, 2, 1, 3, 3, 2, 4, 4, 1, 3, 4, 5, 5, 5, 5, 1, 1])
# print np.hstack((y[:, np.newaxis], X))
# cList = X[:, 3]
# X = X[:, :3]
# 
# (Xtrain, ytrain, Xlo, ylo) = \
#     prepareForTrainingWS(X, y, 2, cList, totalSize = 20)
# print
# print np.hstack((ytrain[:, np.newaxis], Xtrain))
# print
# print np.hstack((ylo[:, np.newaxis], Xlo))
# print getFeatIndices('all')
# print
# print getFeatIndices('allButClrDir')
# print
# print getFeatIndices('allButAnyDir')
