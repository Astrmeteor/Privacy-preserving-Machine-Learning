from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os.path

def readFile(filename):
    # 1- Define parameters (if any)
    filepath = os.path.join(os.path.dirname(__file__), filename) 
    # 2- Read data set
    fH = open(filepath)
    dataMat = np.loadtxt(fH, delimiter=',')
    fH.close()
    return dataMat

# Encoding Types
enc = ["DE", "SUE", "OUE", "SHE", "THE"]
# Pick a dataset
dtID = 4
dNames = ['sample', 'car', 'connect', 'mushroom', 'chess']
th = [0.25, 0, 0, 0, 0]
maxAcc = [0.913, 0.971, 0.7456, 0.9999, 0.9583]

# # X-axis will be epsilon
# fig = plt.figure()
# markers = ['-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D', \
#     '-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D']
# for i, method in enumerate(enc):
#     # retrieve results matrices
#     resMat = readFile(dNames[dtID]+"_encMethod_"+method+".csv")
#     # Plot 
#     plt.plot(resMat[1:, 0], resMat[1:, 1]/100, markers[i], label=method)
# 
# plt.plot([0, 10], [maxAcc[dtID], maxAcc[dtID]], '--') 
# # Grid 
# ax = fig.gca()
# ax.set_xticks(np.arange(0, 10, 1))
# ax.set_yticks(np.arange(0, 1., 0.1))
# plt.rc('grid', linestyle="--")
# plt.grid()
# # Finish the figure
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.xlabel('Epsilon')
# plt.xlim(0.1, 10)
# plt.ylim(0, 1)
# plt.show()
# # # Save
# # filename = os.path.join(os.path.dirname(__file__), 'roc_util_priv.pdf')
# # plt.savefig(filename, bbox_inches='tight')


# ################## THE: different thresholds
# # Encoding Types
# thresholds = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
# 
# # X-axis will be epsilon
# fig = plt.figure()
# markers = ['-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D', \
#     '-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D']
# for i, th in enumerate(thresholds):
#     # retrieve results matrices
#     resMat = readFile("THE\\"+dNames[dtID]+"_encMethod_THE_th_"+str(th)+".csv")
#     # Plot 
#     plt.plot(resMat[1:, 0], resMat[1:, 1]/100, markers[i], label=str(th))
# 
# plt.plot([0, 10], [maxAcc[dtID], maxAcc[dtID]], '--') 
# # Grid 
# ax = fig.gca()
# ax.set_xticks(np.arange(0, 10, 1))
# ax.set_yticks(np.arange(0, 1., 0.1))
# plt.rc('grid', linestyle="--")
# plt.grid()
# # Finish the figure
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.xlabel('Epsilon')
# plt.xlim(0.1, 10)
# plt.ylim(0, 1)
# plt.show()

# ################## DE: using alternative Q: the one in DE equation
# # X-axis will be epsilon
# fig = plt.figure()
# markers = ['-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D', \
#     '-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D']
# for i, dataset in enumerate(dNames):
#     # retrieve results matrices
#     resMat = readFile(dataset+"_encMethod_DE_altQ.csv")
#     # Plot 
#     plt.plot(resMat[1:, 0], resMat[1:, 1]/100, markers[i], label=dataset)
# 
# # plt.plot([0, 10], [maxAcc[dtID], maxAcc[dtID]], '--') 
# # Grid 
# ax = fig.gca()
# ax.set_xticks(np.arange(0, 10, 1))
# ax.set_yticks(np.arange(0, 1., 0.1))
# plt.rc('grid', linestyle="--")
# plt.grid()
# # Finish the figure
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.xlabel('Epsilon')
# plt.xlim(0.1, 10)
# plt.ylim(0, 1)
# plt.show()

# MODIFIED description
th = [0.25, 0, 0, 0, 0]
# X-axis will be epsilon
fig = plt.figure()
markers = ['-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D', \
    '-o', '-v', '-^', '->', '-<', '-8', '-s', '-p', '-*', '-D']
for i, method in enumerate(enc):
    if method == "THE":
        method = method + "_th_" + str(th[dtID])
    # retrieve results matrices
    resMat = readFile(dNames[dtID]+"_encMethod_"+method+".csv")
    # Plot 
    # plt.plot(resMat[1:, 0], resMat[1:, 1]/100, markers[i], label=method)
    # eL = np.array([0, 4, 9, 14, 19, 24, 29, 34, 39, 40, 41, 42]) + 1
    if method == "THE_th_" + str(th[dtID]):
        method = "THE ("+str(th[dtID])+")"
    plt.plot(resMat[1:, 0], resMat[1:, 1]/100, markers[i], label=method)

plt.plot([0, 10], [maxAcc[dtID], maxAcc[dtID]], '--') 
# Grid 
ax = fig.gca()
ax.set_xticks(np.arange(0, 5, 1))
ax.set_yticks(np.arange(0, 1., 0.1))
plt.rc('grid', linestyle="--")
plt.grid()
# Finish the figure
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('Epsilon')
plt.xlim(0.1, 5)
plt.ylim(0, 1.01)
plt.show()
# # Save
# filename = os.path.join(os.path.dirname(__file__), 'roc_util_priv.pdf')
# plt.savefig(filename, bbox_inches='tight')