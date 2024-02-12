from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os.path
from operator import itemgetter

def readFile(filename):
    # 1- Define parameters (if any)
    filepath = os.path.join(os.path.dirname(__file__), filename) 
    # 2- Read data set
    fH = open(filepath)
    dataMat = np.loadtxt(fH, delimiter=',')
    fH.close()
    return dataMat

# Just a dumb way to draw a figure (a quick way)
titles = ['Raw / d1', "PCA / d1", "PCA / d2", "PCA / d4", "DCA / d1", "DCA / d2", "DCA / d4"]
files = ['d1', "PCA_d1", "PCA_d2", "PCA_d4", "DCA_d1", "DCA_d2", "DCA_d4"]

# All Figures
# X-axis will be epsilon
fig = plt.figure()
markers = ['-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D', \
    '-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D']
for i, fID in enumerate(files):
    # retrieve results matrices
    resMat = readFile("mnist_" + fID + "_encMethod_SUE_altQ.csv")
    # Plot 
    plt.plot(resMat[1:, 0], resMat[1:, 1]/100, markers[i], label=titles[i])

# Grid 
ax = fig.gca()
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 1., 0.1))
plt.rc('grid', linestyle="--")
plt.grid()
# Finish the figure
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('Epsilon')
plt.xlim(0.1, 10)
plt.ylim(0, 1)
plt.show()
# # Save
# filename = os.path.join(os.path.dirname(__file__), 'roc_util_priv.pdf')
# plt.savefig(filename, bbox_inches='tight')

# Just PCA vs Raw
ts = titles[0:4]
fs = files[0:4]
# X-axis will be epsilon
fig = plt.figure()
markers = ['-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D', \
    '-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D']
for i, fID in enumerate(fs):
    # retrieve results matrices
    resMat = readFile("mnist_" + fID + "_encMethod_SUE_altQ.csv")
    # Plot 
    plt.plot(resMat[1:, 0], resMat[1:, 1]/100, markers[i], label=ts[i])

# Grid 
ax = fig.gca()
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 1., 0.1))
plt.rc('grid', linestyle="--")
plt.grid()
# Finish the figure
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('Epsilon')
plt.xlim(0.1, 10)
plt.ylim(0, 1)
plt.show()

# Just DCA vs Raw
ts = itemgetter(0, 4, 5, 6)(titles)
fs = itemgetter(0, 4, 5, 6)(files) 
# X-axis will be epsilon
fig = plt.figure()
markers = ['-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D', \
    '-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D']
for i, fID in enumerate(fs):
    # retrieve results matrices
    resMat = readFile("mnist_" + fID + "_encMethod_SUE_altQ.csv")
    # Plot 
    plt.plot(resMat[1:, 0], resMat[1:, 1]/100, markers[i], label=ts[i])

# Grid 
ax = fig.gca()
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 1., 0.1))
plt.rc('grid', linestyle="--")
plt.grid()
# Finish the figure
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('Epsilon')
plt.xlim(0.1, 10)
plt.ylim(0, 1)
plt.show()
