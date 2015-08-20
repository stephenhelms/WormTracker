import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
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
    sigma2 = x.var()
    for i, l in enumerate(lags):
        if l == 0:
            C[i] = 1
        elif l >= x.shape[0]:
            C[i] = ma.masked
        else:
            x0 = x[:-l].copy()
            x1 = x[l:].copy()
            reject = (exclude[l:]-exclude[:-l])>0
            x0[reject] = ma.masked
            x1[reject] = ma.masked
            C[i] = (x0*x1).mean()/sigma2
    return C


def ccf(x, y, lags, exclude=None):
    if exclude is None:
        exclude = np.zeros(x.shape)
    exclude = np.cumsum(exclude.astype(int))

    x = x - x.mean()  # remove mean
    y = y - y.mean()
    if type(lags) is int:
        lags = np.arange(-lags,lags)
    C = ma.zeros((len(lags),))
    sigma2 = x.std()*y.std()
    for i, l in enumerate(lags):
        if l == 0:
            C[i] = (x*y).mean()/sigma2
        else:
            if l > 0:
                x0 = x[:-l].copy()
                y1 = y[l:].copy()
            else:
                x0 = y[:l].copy()
                y1 = x[-l:].copy()
            reject = (exclude[l:]-exclude[:-l])>0
            x0[reject] = ma.masked
            y1[reject] = ma.masked

            C[i] = (x0*y1).mean()/sigma2
    return C


def acv(k, List):
    '''
    Autocovariance
    k is the lag order
    '''
    y = List.copy()
    y = y - y.mean()

    if k == 0:
        return (y*y).mean()
    else:
        return (y[:-k]*y[k:]).mean()


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


def pacfe(p,j,List):
    '''
    Partial autocorrelation function estimates
    p is the order of the AR(p) process
    j is the coefficient in an AR(p) process
    '''
    if p==2 and j==1:
        return (acf(j,List)*(1-acf(p,List)))/(1-(acf(j,List))**2)
    elif p==2 and j==2:
        return (acf(2,List)-(acf(1,List))**2)/(1-(acf(1,List))**2)
    elif p==j and p!=2 and j!=2:
        c=0
        for a in range(1,p):
            c+=pacfe(p-1,a,List)*acf(p-a,List)
        d=0
        for b in range(1,p):
            d+=pacfe(p-1,b,List)*acf(b,List)
        return (acf(p,List)-c)/(1-d)
    else: 
        return pacfe(p-1,j,List)-pacfe(p,p,List)*pacfe(p-1,p-j,List)



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
        elif lag >= x.shape[0]:
            mu[i] = ma.masked
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


def nextpow2(n):
    '''
    Returns the next highest power of 2 from n
    '''
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i


def phaserand(X, independent=False, reduceHighFreqNoise=True):
    '''
    Generates a randomized surrogate dataset for X, preserving linear temporal
    correlations. If independent is False (default), linear correlations
    between columns of x are also preserved.

    If X contains missing values, they are filled with the mean of that
    channel.

    The algorithm works by randomizing the phases in the Fourier domain. For
    non-independent shuffling, the same random phases are used for each
    channel.

    References:
    Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Doyne Farmer, J.
        (1992). Testing for nonlinearity in time series: the method of
        surrogate data. Physica D: Nonlinear Phenomena, 58(1), 77-94.
    Prichard, D. and Theiler, J. (1994) Generating surrogate data for time
        series with several simultaneously measured variables. Phys. Rev.
        Lett. 73(7), 951-954.
    Podobnik, B., Fu, D. F., Stanley, H. E., & Ivanov, P. C. (2007).
        Power-law autocorrelated stochastic processes with long-range
        cross-correlations. The European Physical Journal B, 56(1), 47-52.
    '''
    # Deal with array vs matrix by adding new axis
    if len(X.shape) == 1:
        X = X[:, np.newaxis]


    # Deal with missing data
    if isinstance(X, ma.MaskedArray):
        # truncate all missing data at beginning and end
        idxNotAllMissing = (~np.all(X.mask, axis=1)).nonzero()[0]
        X = X[idxNotAllMissing[0]:idxNotAllMissing[-1], :]
        X = X.filled(X.mean(axis=0))  # fill interior mask with the mean

    # Reduce high-frequency noise by min difference between first and last
    if reduceHighFreqNoise:
        delta = X - X[0, :]
        threshold = 1e-3*np.std(X, axis=0)
        # find last pt in which all the channels are about the same as the beginning
        # and also the index is even
        goodEndPt = np.nonzero((np.all(np.abs(delta) < threshold, axis=1)) &
                               (np.arange(0, X.shape[1]) % 2 == 0))[0][-1]
        if goodEndPt > X.shape[0]/2:  # make sure we keep at least half the data
            X = X[:goodEndPt, :]

    # Fourier transform and extract amplitude and phases
    # The frequencies are shifted so 0 is centered (fftshift)
    N = X.shape[0] #int(nextpow2(X.shape[0]))  # size for FFT
    if N % 2 != 0:
        N = N-1
    h = np.floor(N/2)  # half the length of the data
    Z = np.fft.fft(X, N, axis=0)
    M = np.fft.fftshift(np.abs(Z), axes=0)  # the amplitudes
    phase = np.fft.fftshift(np.angle(Z), axes=0)  # the original phases

    # Randomize the phases. The phases need to be symmetric for postivie and
    # negative frequencies.
    if independent:  # generate random phases for each channel
        randphase = 2.*np.pi*np.random.rand((h-1, X.shape[1]))  # random phases
        newphase = np.zeros((N, X.shape[1]))  # new phases to use
        newphase[0, :] = phase[0, :]  # keep the zero freq (don't know why)
        newphase[1:h, :] = randphase[::-1, :]
        newphase[h, :] = phase[h, :]
        newphase[h+1:, :] = -randphase
    else:  # generate one set of random phases (same as above)
        randphase = 2.*np.pi*np.random.rand(h-1)
        newphase = np.zeros((N, X.shape[1]))
        newphase[0, :] = phase[0, :]
        newphase[1:h, :] = randphase[::-1, np.newaxis]
        newphase[h, :] = phase[h, :]
        newphase[h+1:, :] = -randphase[:, np.newaxis]

    # Reconstruct the signal from the original amplitude and the new phases
    z2 = M*np.exp(newphase*1.j)

    # Return the time-domain signal
    return np.fft.ifft(np.fft.ifftshift(z2, axes=0),
                       axis=0).real.squeeze()
