import os
import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import itertools
import collections
from scipy import stats


def bootstrap(array, nSamples=1000):
    nObserv, nVar = array.shape
    mu = np.zeros((nSamples, nVar))
    replaceIdx = np.random.randint(nObserv, size=(nSamples, 2))
    for i, (iold, inew) in enumerate(replaceIdx):
        resampled = array.copy()
        resampled[iold, :] = resampled[inew, :]
        mu[i, :] = np.mean(resampled, axis=0)

    return (np.mean(mu, axis=0),
            np.percentile(mu, 2.5, axis=0),
            np.percentile(mu, 97.5, axis=0))


def KLDiv(P, Q):
    if P.shape[0] != Q.shape[0]:
        raise Exception()
    if np.any(~np.isfinite(P)) or np.any(~np.isfinite(Q)):
        raise Exception()
    Q = Q / Q.sum()
    P = P / P.sum(axis=0)
    dist = np.sum(P*np.log2(P/Q), axis=0)
    if np.isnan(dist):
        dist = 0
    return dist


def JSDiv(P, Q):
    if P.shape[0] != Q.shape[0]:
        raise Exception()
    Q = Q / Q.sum(axis=0)
    P = P / P.sum(axis=0)
    M = 0.5*(P+Q)
    dist = 0.5*KLDiv(P, M) + 0.5*KLDiv(Q, M)
    return dist
