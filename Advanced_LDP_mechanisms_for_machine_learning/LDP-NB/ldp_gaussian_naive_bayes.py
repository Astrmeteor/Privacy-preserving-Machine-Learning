# -*- coding: utf-8 -*-
from __future__ import division
from numpy.random import laplace,binomial
import numpy as np

class MyGaussianNB:
    def __init__(self, n_labels, n_features):
        self.n_labels = np.array(n_labels)
        self.n_features = np.array(n_features)
        self.mean = np.zeros((n_labels, n_features), dtype=np.float)
        self.var = np.zeros((n_labels, n_features), dtype=np.float)
        self.pi = np.zeros(n_labels, dtype=np.float)

    def predict(self, X):
        joint_log_likelihood = []
        for y, label in enumerate(self.labels):
            jointy = np.log(self.pi[y])
            n_yj = -0.5 * np.sum(np.log(2. * np.pi * self.var[y, :]))
            n_yj -= 0.5 * np.sum( (X-self.mean[y, :])**2 / self.var[y, :] , 1)
            joint_log_likelihood.append(jointy + n_yj)
        jll = np.array(joint_log_likelihood).T
        return self.labels[np.argmax(jll, axis=1)]

    def perturbAlg2LDP(self, X, eps):
        n, d = X.shape
        # Select the columns we will perturb (keep), denoted as "j"s
        js = np.random.randint(0, d, n)
        # Get the values that will be perturbed
        tj = X[np.arange(n), js]
        # Generate the probability for each one
        pArr = ( tj * (np.exp(eps) - 1) + np.exp(eps) + 1 ) / (2 * np.exp(eps) + 2)
        # generate random Bernoulli variables according to these probabilities
        uArr = binomial(1, pArr)
        # Perturbed Data
        pX = np.zeros(X.shape)
        # Set the selected columns to -1 * the value below
        pX[np.arange(n), js] = -1 * d * ( (np.exp(eps) + 1) / (np.exp(eps) - 1) )
        # Turn the (u=1) values to positive by multiplying with -1 
        # (and leaving u=0 values as they were)
        nzInd = np.nonzero(uArr)
        pX[nzInd, js[nzInd]] *= -1
        return pX 

    def perturbBasicLDP(self, X, eps, method):
        n, d = X.shape
        if method == "basicAll":
            return X + laplace(scale=(2 * d / eps), size=(X.shape))
        if method == "basicOne":
            # Select the columns we will perturb (keep), denoted as "j"s
            js = np.random.randint(0, d, n)
            # Perturbed Data
            pX = np.zeros(X.shape)
            # Set the selected columns to -1 * the value below
            pX[np.arange(n), js] = X[np.arange(n), js] + \
                laplace(scale=(2 / eps), size=(n))
            return pX
        return 0

    def fitWithLDP(self, data, labels, eps, method="basicAll"):
        for d in data, labels:
            if not isinstance(d, np.ndarray):
                raise ValueError("The input should be numpy.ndarray");
        if labels.shape[0] != data.shape[0]:
            raise ValueError("The number of data and the number of labels should be same")
        if data.shape != (data.shape[0], self.n_features):
            raise ValueError("The input does not have dimension %s" % self.n_features)

        # the number of training data
        N, d = data.shape
        self.labels = np.unique(labels)
        # # Feature range (to be used for Laplace)
        # featRange = np.max(data, axis=0) - np.min(data, axis=0)
        # featRangeSq = np.max(data**2, axis=0) - np.min(data**2, axis=0)
        # udpate mean and variance of Gaussian
        for y, label in enumerate(self.labels):
            self.pi[y] = np.sum(labels == label) / N 
            # Get data of this class
            X_i = data[labels == label, :]
            # y_i = labels[labels == y]
            # Find mid-point
            mid = int(X_i.shape[0] / 2)
            # Add Laplace noise
            if method == "basicAll" or method == "basicOne":
                pmX_i = self.perturbBasicLDP(X_i[:mid, :], eps, method)
                psX_i = self.perturbBasicLDP(X_i[mid:, :]**2, eps, method)
            elif method == "alg2":
                pmX_i = self.perturbAlg2LDP(X_i[:mid, :], eps)
                psX_i = self.perturbAlg2LDP(X_i[mid:, :]**2, eps)
            # Compute the mean and variance
            if method == "basicAll" or method == "alg2":
                self.mean[y, :] = np.mean(pmX_i, axis=0)
                self.var[y, :] = np.mean(psX_i, axis=0) - self.mean[y, :]**2
            elif method == "basicOne":
                self.mean[y, :] = d * np.mean(pmX_i, axis=0)
                self.var[y, :] = d * np.mean(psX_i, axis=0) - self.mean[y, :]**2
        
        # From Scikit implementation - boost the variances to avoid num. errors
        delta = 1e-9 * np.var(data, axis=0).max()
        # print np.sum(self.var <= 0), "out of", self.var.shape[0]*self.var.shape[1]
        self.var[self.var <= 0] = 1e-1
        self.var[:, :] += delta

    def fit(self, data, labels, cType=0):
        for d in data, labels:
            if not isinstance(d, np.ndarray):
                raise ValueError("The input should be numpy.ndarray");
        if labels.shape[0] != data.shape[0]:
            raise ValueError("The number of data and the number of labels should be same")
        if data.shape != (data.shape[0], self.n_features):
            raise ValueError("The input does not have dimension %s" % self.n_features)

        # the number of training data
        N = data.shape[0] 
        self.labels = np.unique(labels)
        # # Feature range (to be used for Laplace)
        # featRange = np.max(data, axis=0) - np.min(data, axis=0)
        # featRangeSq = np.max(data**2, axis=0) - np.min(data**2, axis=0)
        # udpate mean and variance of Gaussian
        for y, label in enumerate(self.labels):
            self.pi[y] = np.sum(labels == label) / N 
            # Get data of this class
            X_i = data[labels == label, :]
            if type == 0:
                self.mean[y, :] = np.mean(X_i, axis=0) 
                self.var[y, :] = np.var(X_i, axis=0)  
            else:
                # Find mid-point
                mid = int(X_i.shape[0] / 2)
                # Add Laplace noise
                pmX_i = X_i[:mid, :]
                psX_i = X_i[mid:, :]**2
                # Compute the mean and variance
                self.mean[y, :] = np.mean(pmX_i, axis=0)
                self.var[y, :] = np.mean(psX_i, axis=0) - self.mean[y, :]**2 # np.var(pmX_i, axis=0) # 
        
        # From Scikit implementation - boost the variances to avoid num. errors
        delta = 1e-9 * self.var.max()
        # print np.sum(self.var <= 0), "out of", self.var.shape[0]*self.var.shape[1]
        self.var[self.var <= 0] = 1e-1
        self.var[:, :] += delta