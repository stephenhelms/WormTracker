# built-in libraries
import argparse
import os
from copy import deepcopy

# scientific libraries
import numpy as np
import numpy.ma as ma
import yaml
import h5py

# others
from joblib import Parallel, delayed

# wormtracker package
import sys
sys.path.append(r'D:\WormTracker\python')
import wormtracker.analysis as wta
import wormtracker.models as wtm
import wormtracker.config as wtc


def distTo(X, Xr):
    return ma.sqrt(((X-Xr)**2).sum(axis=1))


def normDistToFood(traj):
    X = traj.getMaskedCentroid(traj.X)
    return distTo(X, traj.foodCircle[0:2])/traj.foodCircle[2]


def fractionMissing(traj):
    s = traj.getMaskedPosture(traj.s)  # just a masked variable
    return float(len(s.compressed()))/float(len(s))


def fitModel(traj, mode=None, maxMissing=0.9, excludeEdge=None):
    traj = deepcopy(traj)
    if mode == 'onFood':
        traj.excluded = normDistToFood(traj)>1.1
    if mode == 'offFood':
        traj.excluded = normDistToFood(traj)<=1.1
    if excludeEdge is not None:
            width, height = (traj.h5ref['cropRegion'][-2:]/
                             traj.pixelsPerMicron)
            xsel = np.logical_or(traj.X[:,0] < excludeEdge,
                                 traj.X[:,0] > width - excludeEdge)
            ysel = np.logical_or(traj.X[:,1] < excludeEdge,
                                 traj.X[:,1] > height - excludeEdge)
            sel = np.logical_or(xsel, ysel)
            traj.excluded = np.logical_or(traj.excluded, sel)
    
    m = wtm.Helms2015CentroidModel()
    # check whether there is sufficient datapoints to fit model
    if fractionMissing(traj) > maxMissing:
        p = ma.array(m.toParameterVector()[0], dtype=float)
        p[:] = ma.masked
        return p.filled(np.NAN).astype(float)
    else:
        try:
            m.fit(traj, windowSize=100., plotFit=False)
            return ma.array(m.toParameterVector()[0]).filled(np.NAN).astype(float)
        except Exception as e:
            print 'Error during ' + repr(traj) + repr(e)
            p = ma.array(m.toParameterVector()[0])
            p[:] = ma.masked
            return p.filled(np.NAN).astype(float)


def fitModelFromFile(h5file, strain, wormID, mode=None, maxMissing=0.9,
                     excludeEdge=None):
    with h5py.File(h5file, 'r') as f:
        traj = wta.WormTrajectory(f, strain, wormID)
        return fitModel(traj, mode, maxMissing, excludeEdge=excludeEdge)

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
                       verbose=50)(delayed(fitModelFromFile)(traj.h5obj.filename,
                                                             traj.strain,
                                                             traj.wormID,
                                                             excludeEdge=args.exclude)
                                   for traj in ens)
    #print results
    # keep good results
    T = {traj.wormID: result.tolist()
         for traj, result in zip(ens, results)
         if ~np.isnan(result).all()}

    # save results
    ensName = os.path.splitext(args.ens)[0]
    with open('{0}_cmf.yml'.format(ensName), 'wb') as f:
        yaml.dump(T, f)

if __name__ == "__main__":
    os.chdir(r'D:\Stephen')
    main(sys.argv)
