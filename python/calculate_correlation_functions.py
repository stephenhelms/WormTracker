# built-in libraries
import argparse
import os

# scientific libraries
import numpy as np
import numpy.ma as ma
import h5py

# others
from joblib import Parallel, delayed

import sys

# my statistics library
sys.path.append(r'D:\stephen\code\statstools')
import stats
import tsstats

# wormtracker package
sys.path.append(r'D:\WormTracker\python')
import wormtracker.analysis as wta
import wormtracker.models as wtm
import wormtracker.config as wtc

lags = np.arange(1150)
x = np.linspace(0,500,1000)


def fractionMissing(traj):
    s = traj.getMaskedPosture(traj.s)  # just a masked variable
    return float(len(s.compressed()))/float(len(s))


def speedWithoutReversals(traj):
    s = traj.getMaskedPosture(traj.s)
    s[traj.nearRev] = ma.masked
    return s


def angleToVector(alpha):
    return ma.array([ma.cos(alpha), ma.sin(alpha)]).T


def calcCorrelationFunctionsFromFile(h5file, strain, wormID, maxMissing=0.5,
                     excludeEdge=None):
    with h5py.File(h5file, 'r') as f:
        traj = wta.WormTrajectory(f, strain, wormID)
        if excludeEdge is not None:
            width, height = (traj.h5ref['cropRegion'][-2:]/
                             traj.pixelsPerMicron)
            xsel = np.logical_or(traj.X[:,0] < excludeEdge,
                                 traj.X[:,0] > width - excludeEdge)
            ysel = np.logical_or(traj.X[:,1] < excludeEdge,
                                 traj.X[:,1] > height - excludeEdge)
            sel = np.logical_or(xsel, ysel)
            traj.excluded = np.logical_or(traj.excluded, sel)
        traj.identifyReversals()

        return (msd(traj, maxMissing), vacf(traj, maxMissing), s_acf(traj, maxMissing),
                psi_acf(traj, maxMissing), dpsi_acf(traj, maxMissing),
                psi_trend(traj, maxMissing), s_dist(traj, maxMissing))


def msd(traj, maxMissing):
    # calculate the MSD
    D = ma.array([wta.meanSquaredDisplacement(trajw.getMaskedPosture(trajw.X), lags, exclude=trajw.excluded)
                  for trajw in traj.asWindows(100.)
                  if fractionMissing(trajw)>maxMissing])
    D[np.isnan(D)] = ma.masked
    D = D.mean(axis=0)
    if len(D) == 0:
        D = ma.zeros((len(lags),))
        D[:] = ma.masked
    return D


def vacf(traj, maxMissing):
    C = ma.array([tsstats.dotacf(trajw.getMaskedPosture(trajw.v), lags, exclude=trajw.excluded)
                  for trajw in traj.asWindows(100.)
                  if fractionMissing(trajw)>maxMissing])
    C[np.isnan(C)] = ma.masked
    C = C.mean(axis=0)
    if len(C) == 0:
        C = ma.zeros((len(lags),))
        C[:] = ma.masked
    return C


def s_acf(traj, maxMissing):
    sigma2 = ma.array([speedWithoutReversals(trajw).var()
                       for trajw in traj.asWindows(100.)
                       if fractionMissing(trajw)>maxMissing])
    C = ma.array([tsstats.acf(speedWithoutReversals(trajw), lags, exclude=trajw.excluded)
                  for trajw in traj.asWindows(100.)
                  if fractionMissing(trajw)>maxMissing])
    C[np.isnan(C)] = ma.masked
    C = (sigma2[:,np.newaxis]*C).mean(axis=0)
    if len(C) == 0:
        C = ma.zeros((len(lags),))
        C[:] = ma.masked
    return C


def s_dist(traj, maxMissing):
    import scipy.stats as ss
    s = traj.getMaskedPosture(traj.s)
    s[traj.nearRev] = ma.masked
    s = s.compressed()
    if len(s) > 100:
        kd = ss.gaussian_kde(s)
        return ma.array(kd.evaluate(x))
    else:
        y = ma.zeros((len(x),))
        y[:] = ma.masked
        return y


def psi_acf(traj, maxMissing):
    C = ma.array([tsstats.dotacf(angleToVector(trajw.getMaskedPosture(trajw.psi)), lags, exclude=trajw.excluded)
                  for trajw in traj.asWindows(100.)
                  if fractionMissing(trajw)>maxMissing])
    C[np.isnan(C)] = ma.masked
    C = C.mean(axis=0)
    if len(C) == 0:
        C = ma.zeros((len(lags),))
        C[:] = ma.masked
    return C


def dpsi_acf(traj, maxMissing):
    C = ma.array([tsstats.dotacf(angleToVector(trajw.getMaskedPosture(trajw.dpsi)), lags, exclude=trajw.excluded)
                  for trajw in traj.asWindows(100.)
                  if fractionMissing(trajw)>maxMissing])
    C[np.isnan(C)] = ma.masked
    C = C.mean(axis=0)
    if len(C) == 0:
        C = ma.zeros((len(lags),))
        C[:] = ma.masked
    return C


def psi_trend(traj, maxMissing):    
    D = ma.array([tsstats.drift(tsstats.unwrapma(trajw.getMaskedPosture(trajw.psi)), lags, exclude=trajw.excluded)
                  for trajw in traj.asWindows(100.)
                  if fractionMissing(trajw)>maxMissing])
    D[np.isnan(D)] = ma.masked
    D = ma.abs(D).mean(axis=0)
    if len(D) == 0:
        D = ma.zeros((len(lags),))
        D[:] = ma.masked
    return D


def main(argv):
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--ens')
    parser.add_argument('-d', '--dir')
    parser.add_argument('-x', '--exclude', type=float)
    args = parser.parse_args()

    # switch directory if needed
    if args.dir:
        os.chdir(args.dir)

    # load target ensemble
    ens = wtc.loadEnsemble(args.ens)

    results = Parallel(n_jobs=-1,
                       verbose=50)(delayed(calcCorrelationFunctionsFromFile)(traj.h5obj.filename,
                                                                             traj.strain,
                                                                             traj.wormID,
                                                                             excludeEdge=args.exclude)
                                   for traj in ens)
    
    # save results
    ensName = os.path.splitext(args.ens)[0]
    f = h5py.File('{0}_corrfun.h5'.format(ensName), 'w')
    f.create_dataset('taus', (len(lags),), dtype=float)
    f['taus'][...] = lags/11.5
    f.create_dataset('x_speed', (len(x),), dtype=float)
    f['x_speed'][...] = x

    f.create_dataset('wormID', (len(ens),), dtype='S10')
    f.create_dataset('strain', (len(ens),), dtype='S10')
    f.create_dataset('MSD', (len(ens), len(lags)), dtype=float)
    f.create_dataset('VACF', (len(ens), len(lags)), dtype=float)
    f.create_dataset('s_ACF', (len(ens), len(lags)), dtype=float)
    f.create_dataset('psi_ACF', (len(ens), len(lags)), dtype=float)
    f.create_dataset('dpsi_ACF', (len(ens), len(lags)), dtype=float)
    f.create_dataset('psi_trend', (len(ens), len(lags)), dtype=float)
    for i in xrange(len(ens)):
        result = results[i]
        traj = ens[i]
        f['strain'][i] = np.string_(traj.strain)
        f['wormID'][i] = np.string_(traj.wormID)
        f['MSD'][i] = result[0].filled(np.nan)
        f['VACF'][i] = result[1].filled(np.nan)
        f['s_ACF'][i]= result[2].filled(np.nan)
        f['psi_ACF'][i] = result[3].filled(np.nan)
        f['dpsi_ACF'][i] = result[4].filled(np.nan)
        f['psi_trend'][i] = result[5].filled(np.nan)
        f['s_dist'][i]= result[6].filled(np.nan)

    f.close()
    

if __name__ == "__main__":
    os.chdir(r'D:\Stephen')
    main(sys.argv)
