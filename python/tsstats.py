import os
import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import collections
from scipy import stats


def acf(x, lags=500):
    # from stackexchange
    x = x - x.mean()  # remove mean
    if type(lags) is int:
        lags = range(lags)

    C = ma.zeros((len(lags), 1))
    for i, l in enumerate(lags):
        if l == 0:
            C[i] = 1
        else:
            C[i] = ma.corrcoef(x[:-l], x[l:])[0, 1]
    return C


def circacf(x, lags=500):
    if type(lags) is int:
        lags = xrange(1, lags)

    return np.array([1] +
                    [np.mean(np.cos(x[lag:]-x[:-lag]))
                     for lag in lags])


def dotacf(x, lags=500):
    if type(lags) is int:
        lags = xrange(lags)
    C = ma.zeros((len(lags),))
    for i, l in enumerate(lags):
        if l == 0:
            C[i] = (x*x).sum(axis=1).mean()
        else:
            C[i] = (x[l:, :]*x[:-l, :]).sum(axis=1).mean()
    return C
