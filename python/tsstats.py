import os
import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import collections
from scipy import stats


def acf(x, lags=500, exclude=None):
    if exclude is None:
        exclude = np.zeros(x.shape)
    exclude = np.cumsum(exclude.astype(int))

    # from stackexchange
    x = x - x.mean()  # remove mean
    if type(lags) is int:
        lags = range(lags)

    C = ma.zeros((len(lags),))
    for i, l in enumerate(lags):
        if l == 0:
            C[i] = 1
        else:
            x0 = x[:-l].copy()
            x1 = x[l:].copy()
            reject = (exclude[l:]-exclude[:-l])>0
            x0[reject] = ma.masked
            x1[reject] = ma.masked
            C[i] = ma.corrcoef(x0, x1)[0, 1]
    return C


def dotacf(x, lags=500, exclude=None):
    if exclude is None:
        exclude = np.zeros(x.shape)
    exclude = np.cumsum(exclude.astype(int))

    if type(lags) is int:
        lags = xrange(lags)
    C = ma.zeros((len(lags),))
    for i, l in enumerate(lags):
        if l == 0:
            C[i] = (x*x).sum(axis=1).mean()
        else:
            x0 = x[:-l, :].copy()
            x1 = x[l:, :].copy()
            reject = (exclude[l:]-exclude[:-l])>0
            x0[reject, :] = ma.masked
            x1[reject, :] = ma.masked
            C[i] = (x0*x1).sum(axis=1).mean()
    return C


def drift(x, lags=500, exclude=None):
    if exclude is None:
        exclude = np.zeros(x.shape)
    exclude = np.cumsum(exclude.astype(int))

    if type(lags) is int:
        lags = xrange(lags)
    mu = ma.zeros((len(lags),))
    for i, lag in enumerate(lags):
        if lag==0:
            mu[i] = 0
        else:
            x0 = x[lag:].copy()
            x1 = x[:-lag].copy()
            reject = (exclude[lag:]-exclude[:-lag])>0
            x0[reject] = ma.masked
            x1[reject] = ma.masked
            displacements = x0 - x1
            mu[i] = displacements.mean()
    return mu


def unwrapma(x):
    # Adapted from numpy unwrap, this version ignores missing data
    idx = ma.array(np.arange(0,x.shape[0]), mask=x.mask)
    idxc = idx.compressed()
    xc = x.compressed()
    dd = np.diff(xc)
    ddmod = np.mod(dd+np.pi, 2*np.pi)-np.pi
    ddmod[(ddmod==-np.pi) & (dd > 0)] = np.pi
    phc_correct = ddmod - dd
    phc_correct[np.abs(dd)<np.pi] = 0
    ph_correct = np.zeros(x.shape)
    ph_correct[idxc[1:]] = phc_correct
    up = x + ph_correct.cumsum()
    return up
