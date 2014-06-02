import os
import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import itertools
import collections
import cv2
import wormtracker.wormimageprocessor as wip
import multiprocessing as multi
from numba import jit


def configureMatplotLibStyle():
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.labelsize'] = 'x-large'
    mpl.rcParams['axes.titlesize'] = 'x-large'
    mpl.rcParams['grid.color'] = (0.5, 0.5, 0.5)
    mpl.rcParams['legend.fontsize'] = 'medium'
    mpl.rcParams['legend.numpoints'] = 1
    mpl.rcParams['legend.frameon'] = False


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


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def acf(x, lags=500):
    # from stackexchange
    x = x - x.mean() # remove mean
    if type(lags) is int:
        lags = range(1, lags)

    return ma.array([1] +
                    [ma.corrcoef(x[:-i], x[i:])[0, 1]
                     for i in lags])


def circacf(x, lags=500):
    if type(lags) is int:
        lags = xrange(1, lags)

    return np.array([1] +
                    [np.mean(np.cos(x[lag:]-x[:-lag]))
                     for lag in lags])


def dotacf(x, lags=500):
    if type(lags) is int:
        lags = xrange(lags)
    return [np.mean(np.vdot(x[l:, :], x[:-l, :]))
            for l in lags]


class WormTrajectory:
    filterByWidth = False

    def __init__(self, h5obj, strain, wormID, videoFilePath=None):
        self.firstFrame = None
        self.h5obj = h5obj
        self.h5ref = self.h5obj['worms'][strain][wormID]
        self.strain = strain
        self.wormID = wormID
        self.frameRate = self.h5obj['/video/frameRate'][0]
        self.pixelsPerMicron = self.h5obj['/video/pixelsPerMicron'][0]
        self.foodCircle = self.h5ref['foodCircle'][...] / self.pixelsPerMicron
        self.t = self.h5ref['time']
        self.maxFrameNumber = self.t.shape[0]
        self.X = self.h5ref['X']
        self.Xhead = self.h5ref['Xhead']
        self.v = self.h5ref['v']
        self.s = self.h5ref['s']
        self.phi = self.h5ref['phi']
        self.psi = self.h5ref['psi']
        self.dpsi = self.h5ref['dpsi']
        self.Ctheta = self.h5ref['Ctheta']
        self.ltheta = self.h5ref['ltheta']
        self.vtheta = self.h5ref['vtheta']
        self.length = self.h5ref['avgLength']
        self.width = self.h5ref['avgWidth']
        self.badFrames = self.h5ref['badFrames'][...]
        self.allCentroidMissing = np.all(self.badFrames)

        if videoFilePath is not None:
            videoFile = self.h5obj['/video/videoFile'][0]
            self.videoFile = os.path.join(videoFilePath, videoFile)
        else:
            self.videoFile = None

        self.skeleton = self.h5ref['skeletonSpline']
        self.posture = self.h5ref['posture']
        self.orientationFixed = self.h5ref['orientationFixed'][...]
        self.allPostureMissing = np.all(np.logical_not(self.orientationFixed))

    def readFirstFrame(self):
        if self.videoFile is None:
            self.firstFrame = None
            return
        video = cv2.VideoCapture()
        if video.open(self.videoFile):
            success, firstFrame = video.read()
            if not success:
                raise Exception("Couldn't read video")
            else:
                firstFrameChannels = cv2.split(firstFrame)
                frame = firstFrameChannels[0]
                crop = self.h5ref['cropRegion'][...]
                frame = wip.cropImageToRegion(frame, crop)
                ip = self.getImageProcessor()
                frame = ip.applyBackgroundFilter(frame)
                self.firstFrame = cv2.normalize(frame,
                                                alpha=0,
                                                beta=255,
                                                norm_type=cv2.NORM_MINMAX)
        else:
            raise Exception("Couldn't open video")

    def getImageProcessor(self):
        ip = wip.WormImageProcessor()
        ip.backgroundDiskRadius = \
            self.h5obj['/video/backgroundDiskRadius'][0]
        ip.pixelSize = self.h5obj['/video/pixelsPerMicron'][0]
        ip.threshold = self.h5obj['/video/threshold'][0]
        ip.wormDiskRadius = self.h5obj['/video/wormDiskRadius'][0]
        ip.wormAreaThresholdRange = \
            self.h5obj['/video/wormAreaThresholdRange'][...]
        # length
        # width
        return ip

    def getMaskedCentroid(self, data):
        data = ma.array(data)
        sel = self.badFrames
        data[sel, ...] = ma.masked
        data[np.isnan(data)] = ma.masked
        return data

    def getMaskedPosture(self, data):
        data = ma.array(data)
        sel = np.logical_or(np.logical_not(self.orientationFixed),
                            self.badFrames)
        data[sel, ...] = ma.masked
        data[np.isnan(data)] = ma.masked
        return data

    def plotTrajectory(self, color='k', showFrame=True, showPlot=True):
        if showFrame and self.firstFrame is not None:
            plt.imshow(self.firstFrame, plt.gray(),
                       origin='lower',
                       extent=(0,
                               self.firstFrame.shape[1]/self.pixelsPerMicron,
                               0,
                               self.firstFrame.shape[0]/self.pixelsPerMicron))
            plt.hold(True)
        X = self.getMaskedCentroid(self.X)
        plt.scatter(X[:, 0], X[:, 1], c=color, s=10)
        plt.hold(True)
        if self.foodCircle is not None:
            circle = plt.Circle(self.foodCircle[0:2],
                                radius=self.foodCircle[-1],
                                color='r', fill=False)
            plt.gca().add_patch(circle)
        plt.xlim((0, 10000))
        plt.ylim((0, 10000))
        plt.xlabel('x (um)')
        plt.xlabel('y (um)')
        plt.gca().set_aspect('equal')
        if showPlot:
            plt.show()

    def plotSpeed(self, showPlot=True):
        s = self.getMaskedCentroid(self.s)
        plt.plot(self.t, s, 'k.')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (um/s)')
        if showPlot:
            plt.show()

    def plotBearing(self, showPlot=True):
        phi = self.getMaskedCentroid(self.phi)
        plt.plot(self.t, phi/np.pi, 'k.')
        plt.xlabel('Time (s)')
        plt.ylabel('Bearing ($\pi$ rad)')
        if showPlot:
            plt.show()

    def getSpeedDistribution(self, bins=None):
        if bins is None:
            bins = np.ceil(np.sqrt(np.sum(np.logical_not(self.badFrames))))

        s = self.getMaskedCentroid(self.s)
        out = np.histogram(s.compressed(), bins,
                           density=True)
        return out

    def plotSpeedDistribution(self, bins=None, color='k', showPlot=True):
        if bins is None:
            bins = np.ceil(np.sqrt(np.sum(np.logical_not(self.badFrames))))
        s = self.getMaskedCentroid(self.s)
        plt.hist(s.compressed(), bins, normed=True, facecolor=color)
        plt.xlabel('Speed (um/s)')
        plt.ylabel('Probability')
        if showPlot:
            plt.show()

    def getSpeedAutocorrelation(self, maxT=100):
        n = int(np.round(maxT*self.frameRate))
        tau = range(n)/self.frameRate
        s = self.getMaskedCentroid(self.s)
        C = acf(s, n)
        return tau, C

    def plotSpeedAutocorrelation(self, maxT=100, color='k', showPlot=True):
        tau, C = self.getSpeedAutocorrelation(maxT)
        plt.plot(tau, C, '-', color=color)
        plt.xlabel(r'$\tau (s)$')
        plt.ylabel(r'$\langle s(t) \cdot s(t+\tau)\rangle$')
        if showPlot:
            plt.show()

    def getBearingAutocorrelation(self, maxT=100):
        n = int(np.round(maxT*self.frameRate))
        tau = range(n)/self.frameRate
        psi = self.getMaskedPosture(self.psi)
        C = circacf(psi, n)
        return tau, C

    def plotBearingAutocorrelation(self, maxT=100, color='k', showPlot=True):
        tau, C = self.getBearingAutocorrelation(maxT)
        plt.semilogx(tau, C, '-', color=color)
        plt.xlabel(r'$\log \tau / (s)$')
        plt.ylabel(r'$\langle \cos\left[\psi(t)-\psi(t+\tau)\right]\rangle$')
        if showPlot:
            plt.show()

    def getMeanSquaredDisplacement(self, tau=None):
        if tau is None:
            tau = np.logspace(-1, 3, 200)

        lags = np.round(tau*self.frameRate)
        Sigma = ma.zeros((200,))
        X = self.getMaskedCentroid(self.X)
        for i, lag in enumerate(lags):
            displacements = X[lag:, :] - X[:-lag, :]
            Sigma[i] = np.mean(np.log10(np.sum(displacements**2, axis=1)))

        return (tau, Sigma)

    def plotMeanSquaredDisplacement(self, tau=None, showPlot=True):
        tau, Sigma = self.getMeanSquaredDisplacement(tau)
        plt.plot(np.log10(tau), Sigma, 'k.')
        plt.xlabel(r'log $\tau$ \ (s)')
        plt.ylabel(r'log $\langle \| x(t) - x(t-\tau) \|^2 \rangle$ (um^2)')
        if showPlot:
            plt.show()

    def plotPosturalCovariance(self, showPlot=True):
        if self.Ctheta is None:
            return
        plt.imshow(self.Ctheta, plt.get_cmap('PuOr'))
        plt.clim((-0.5, 0.5))
        plt.colorbar()
        plt.grid(False)
        if showPlot:
            plt.show()

    def plotPosturalModeDistribution(self, color='k', showPlot=True):
        if self.Ctheta is None:
            return
        plt.plot(np.cumsum(self.ltheta)/np.sum(self.ltheta), '.-', color=color,
                 label='{0} {1}'.format(self.strain, self.wormID))
        plt.xlabel('Postural Mode')
        plt.ylabel('%% Variance')
        plt.ylim((0, 1))
        if showPlot:
            plt.show()

    def plotPosturalTimeSeries(self, postureVec, color='k', showPlot=True):
        if self.Ctheta is None:
            return
        if isinstance(postureVec, int):
            postureVec = self.vtheta[:, postureVec]
        posture = self.getMaskedPosture(self.posture)
        A = np.dot(posture, postureVec)
        missing = np.any(posture.mask, axis=1)
        A[missing] = ma.masked
        plt.plot(self.t, A, '.', color=color,
                 label='{0} {1}'.format(self.strain, self.wormID))
        if showPlot:
            plt.show()

    def plotPosturalPhaseSpace(self, postureVec1, postureVec2, color='k',
                               showPlot=True):
        if self.Ctheta is None:
            return
        if isinstance(postureVec1, int):
            postureVec1 = self.vtheta[:, postureVec1]
        if isinstance(postureVec2, int):
            postureVec2 = self.vtheta[:, postureVec2]
        posture = self.getMaskedPosture(self.posture)
        missing = np.any(posture.mask, axis=1)
        A = np.dot(posture, postureVec1)
        A[missing] = ma.masked
        B = np.dot(posture, postureVec2)
        B[missing] = ma.masked
        plt.scatter(A, B, marker='.', c=color, s=5)
        if showPlot:
            plt.show()


class WormTrajectoryStateImage:
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.t = self.trajectory.t
        self.badFrames = self.trajectory.badFrames
        self.X = self.trajectory.getMaskedCentroid(self.trajectory.X)
        self.Xhead = self.trajectory.getMaskedPosture(self.trajectory.Xhead)
        self.skeleton = ma.array(self.trajectory.skeleton[:, 1:-1, :])
        self.skeleton[~self.trajectory.orientationFixed, :, :] = ma.masked
        self.posture = self.trajectory.getMaskedPosture(self.trajectory.posture)
        self.pixelsPerMicron = self.trajectory.pixelsPerMicron
        self.frameRate = self.trajectory.frameRate

    def initialView(self):
        self.axTraj = plt.subplot(1, 2, 1)
        if self.trajectory.foodCircle is not None:
            self.foodCircle = plt.Circle(self.trajectory.foodCircle[0:2],
                                radius=self.trajectory.foodCircle[-1],
                                color='r', fill=False)
            plt.gca().add_patch(self.foodCircle)
        self.trajLine, = self.axTraj.plot([], [], 'k.')
        plt.hold(True)
        self.trajPoint, = self.axTraj.plot([], [], 'ro')
        plt.xlim((0, 10000))
        plt.xlabel('x (um)')
        plt.ylim((0, 10000))
        plt.ylabel('y (um)')
        self.axTraj.set_aspect(1)

        self.axWorm = plt.subplot(1, 2, 2)
        self.imWorm = self.axWorm.imshow(np.zeros((1, 1)), plt.gray(),
                                         origin='lower',
                                         interpolation='none',
                                         vmin=0, vmax=255)
        plt.hold(True)
        self.skelLine, = self.axWorm.plot([], [], 'c-')
        self.postureSkel = self.axWorm.scatter([], [], c=[],
                                               cmap=plt.get_cmap('PuOr'),
                                               s=100,
                                               vmin=-3,
                                               vmax=3)
        self.centroid, = self.axWorm.plot([], [], 'ro')
        self.head, = self.axWorm.plot([], [], 'rs')
        plt.xticks([])
        plt.yticks([])
        #self.

    def plot(self, frameNumber):
        # trajectory plot
        self.trajLine.set_xdata(self.X[:frameNumber, 0])
        self.trajLine.set_ydata(self.X[:frameNumber, 1])
        self.trajPoint.set_xdata(self.X[frameNumber, 0])
        self.trajPoint.set_ydata(self.X[frameNumber, 1])
        # worm plot
        im = self.getWormImage(frameNumber)
        self.imWorm.set_array(im)
        self.imWorm.set_extent((0, im.shape[1], 0, im.shape[0]))
        bb = self.trajectory.h5ref['boundingBox'][frameNumber,
                                                  :2]
        skel = self.trajectory.h5ref['skeleton'][frameNumber, :, :]
        empty = np.all(skel == 0, axis=1)
        self.skelLine.set_xdata(skel[~empty, 1])
        self.skelLine.set_ydata(skel[~empty, 0])
        self.postureSkel.set_offsets(np.fliplr(self.skeleton[frameNumber, :, :]))
        self.postureSkel.set_array(self.posture[frameNumber, :])
        self.postureSkel.set_clim(-1, 1)
        self.postureSkel.set_cmap(plt.get_cmap('PuOr'))
        self.centroid.set_xdata(self.X[frameNumber, 1]*self.pixelsPerMicron - bb[1])
        self.centroid.set_ydata(self.X[frameNumber, 0]*self.pixelsPerMicron - bb[0])
        self.head.set_xdata(self.Xhead[frameNumber, 1]*self.pixelsPerMicron - bb[1])
        self.head.set_ydata(self.Xhead[frameNumber, 0]*self.pixelsPerMicron - bb[0])
        plt.title(str(frameNumber/self.frameRate) + ' s')

    def getWormImage(self, frameNumber):
        if not self.badFrames[frameNumber]:
            bb = self.trajectory.h5ref['boundingBox'][frameNumber,
                                                      2:]
            if all(bb == 0):
                return np.zeros((100,100))
            im = self.trajectory.h5ref['grayWormImage'][frameNumber,
                                                        :bb[1],
                                                        :bb[0]]
            im = cv2.normalize(im, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX)
        else:
            im = np.zeros((100, 100))
        return im

    def getAnimation(self, frames=None, interval=None, figure=None):
        if figure is None:
            figure = plt.figure()
        if frames is None:
            good = (~self.badFrames).nonzero()[0]
            frames = xrange(good[0], good[-1])
        if interval is None:
            interval = 1000/self.frameRate
        return animation.FuncAnimation(figure, self.plot,
                                       frames=frames,
                                       init_func=self.initialView,
                                       interval=interval)

    def showAnimation(self, frames=None, interval=None):
        self.getAnimation(frames=frames, interval=interval)
        plt.show()


class WormTrajectoryEnsemble:
    def __init__(self, trajectoryIter=None, name=None, nameFunc=None):
        if any(not isinstance(it, WormTrajectory) for it in trajectoryIter):
            raise TypeError('A trajectory ensemble must contain ' +
                            'WormTrajectory objects.')
        self._trajectories = list(trajectoryIter)
        self.name = name
        if nameFunc is None:
            nameFunc = lambda t: t.strain + ' ' + t.wormID
        self.nameFunc = nameFunc

    def __iter__(self):
        for traj in self._trajectories:
            yield traj

    #  TODO: Implement
    #  __add__(), __radd__(), __iadd__(), __mul__(), __rmul__() and __imul__()

    def __add__(self, other):
        if isinstance(other, WormTrajectory):
            return WormTrajectoryEnsemble(self + other)
        elif isinstance(other, WormTrajectoryEnsemble):
            return WormTrajectoryEnsemble(self + other._trajectories)
        elif (isinstance(other, collections.Iterable) and
              all([isinstance(t, WormTrajectoryEnsemble)
                   for t in other])):
            return WormTrajectoryEnsemble(self + other)
        else:
            raise ValueError('{0} is not a valid'.format(str(type(other))) +
                             ' type to add to a WormTrajectoryEnsemble')

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, WormTrajectory):
            self.append(other)
        elif isinstance(other, WormTrajectoryEnsemble):
            return self.extend(other._trajectories)
        elif (isinstance(other, collections.Iterable) and
              all([isinstance(t, WormTrajectory)
                   for t in other])):
            return self.extend(other)
        else:
            raise ValueError('{0} is not a valid'.format(str(type(other))) +
                             ' type to add to a WormTrajectoryEnsemble')

    def __isub__(self, other):
        if isinstance(other, WormTrajectory):
            self.remove(other)
        elif isinstance(other, WormTrajectoryEnsemble):
            for t in other:
                self.remove(t)
        elif (isinstance(other, collections.Iterable) and
              all([isinstance(t, WormTrajectory)
                   for t in other])):
            for t in other:
                self.remove(t)
        else:
            raise ValueError('{0} is not a valid'.format(str(type(other))) +
                             ' type to remove from a WormTrajectoryEnsemble')

    def __eq__(self, other):
        return all([(traj in other) for traj in self])

    def __ne__(self, other):
        return any([(traj not in other) for traj in self])

    def __getitem__(self, key):
        return self._trajectories[key]

    def __setitem__(self, key, value):
        self._trajectories[key] = value

    def __delitem__(self, key):
        del self._trajectories[key]

    def __contains__(self, value):
        return value in self._trajectories

    def __len__(self):
        return len(self._trajectories)

    def __getslice__(self, i, j):
        return WormTrajectoryEnsemble(self._trajectories[i:j])

    def __setslice__(self, i, j, sequence):
        self._trajectories[i:j] = sequence

    def __delslice__(self, i, j):
        del self._trajectories[i:j]

    def append(self, item):
        self._trajectories.append(item)

    def count(self):
        return self._trajectories.count()

    def index(self, value):
        return self._trajectories.index(value)

    def extend(self, iter):
        self._trajectories.extend(iter)

    def insert(self, index, item):
        self._trajectories.insert(index, item)

    def pop(self, index):
        self._trajectories.pop([index])

    def remove(self, item):
        self._trajectories.remove(item)

    def reverse(self):
        self._trajectories.reverse()

    def sort(self, cmp=None, key=None):
        if cmp is None and key is None:
            key = lambda t: int(t.wormID)
        self._trajectories.sort(cmp=cmp, key=key)

    def splitByStrain(self):
        strains = set([traj.strain for traj in self])
        return {strain: WormTrajectoryEnsemble([traj for traj in self
                                                if traj.strain == strain])
                for strain in strains}

    def readFirstFrameAll(self):
        for t in self:
            t.readFirstFrame()

    def calculatePosturalMeasurements(self):
        posture = []
        for traj in self:
            posturea = traj.getMaskedPosture(traj.posture)
            missing = np.any(posturea.mask, axis=1)
            if np.all(missing):
                continue
            else:
                posture.append(posturea[~missing, :])   
        if len(posture) > 0:
            posture = np.concatenate(posture).T
            self.Ctheta = np.cov(posture)
            self.ltheta, self.vtheta = LA.eig(self.Ctheta)
        else:
            self.Ctheta = None
            self.ltheta = None
            self.vtheta = None

    def ensembleAverage(self, compFunc, nSamples=1000):
        samples = np.array([compFunc(traj) for traj in self])
        return bootstrap(samples, nSamples)

    def tilePlots(self, plotFunc, ni=4, nj=4):
        for i, t in enumerate(self):
            plt.subplot(ni, nj, i+1)
            plotFunc(t)
            plt.title(self.nameFunc(t))
        plt.show()

    def getMedianSpeed(self):
        func = lambda t: np.array([ma.median(t.getMaskedCentroid(t.s))])
        return self.ensembleAverage(func)

    def getMeanSpeed(self):
        func = lambda t: np.array([ma.mean(t.getMaskedCentroid(t.s))])
        return self.ensembleAverage(func)

    def plotSpeedDistribution(self, bins=None, color='k', showPlot=True):
        if bins is None:
            bins = np.linspace(0, 500, 200)
        p, pl, pu = self.ensembleAverage(lambda x: x.getSpeedDistribution(bins)[0])
        centers = [(x1+x2)/2 for x1, x2 in pairwise(bins)]
        plt.plot(centers, p, '.-', color=color, label=self.name)
        plt.hold(True)
        plt.fill_between(centers, pl, pu, facecolor=color, alpha=0.3)
        plt.xlabel('Speed (um/s)')
        plt.ylabel('Probability')
        plt.xlim([0,max(bins)])
        if showPlot:
            plt.show()

    def plotSpeedDistributions(self, bins=None, showPlot=True):
        if bins is None:
            bins = np.linspace(0, 500, 200)
        centers = [(x1+x2)/2 for x1, x2 in pairwise(bins)]
        for i, t in enumerate(self):
            color = plt.cm.jet(float(i)/float(len(self)-1))
            p = t.getSpeedDistribution(bins)[0]
            plt.plot(centers,p,'.-',color=color,
                     label=self.nameFunc(t))
            plt.hold(True)
        plt.xlabel('Speed (um/s)')
        plt.ylabel('Probability')
        plt.xlim([0, max(bins)])
        plt.legend()
        if showPlot:
            plt.show()

    def plotSpeedAutocorrelation(self, maxT=100, color='k', showPlot=True):
        # assume all same frame rate
        n = int(np.round(maxT*self[0].frameRate))
        tau = range(n)/self[0].frameRate
        C, Cl, Cu = self.ensembleAverage(lambda x: x.getSpeedAutocorrelation(maxT)[1])
        plt.plot(tau, C, '.-', color=color, label=self.name)
        plt.fill_between(tau, Cl, Cu, facecolor=color, alpha=0.3)
        plt.xlabel(r'$\tau (s)$')
        plt.ylabel(r'$\langle s(t) \cdot s(t+\tau)\rangle$')
        # TODO: smart xlim
        if showPlot:
            plt.show()

    def plotBearingAutocorrelation(self, maxT=100, color='k', showPlot=True):
        # assume all same frame rate
        n = int(np.round(maxT*self[0].frameRate))
        tau = range(n)/self[0].frameRate
        C, Cl, Cu = self.ensembleAverage(lambda x: x.getBearingAutocorrelation(maxT)[1])
        plt.semilogx(tau, C, '.-', color=color, label=self.name)
        plt.fill_between(tau, Cl, Cu, facecolor=color, alpha=0.3)
        plt.xlabel(r'$\log \tau / (s)$')
        plt.ylabel(r'$\langle s(t) \cdot s(t+\tau)\rangle$')
        # TODO: smart xlim
        if showPlot:
            plt.show()

    def plotMeanSquaredDisplacement(self, tau=None, color='k', showPlot=True):
        tau = np.logspace(-1,3,200)
        S, Sl, Su = self.ensembleAverage(lambda x: x.getMeanSquaredDisplacement(tau)[1])
        log_tau = np.log10(tau)
        plt.plot(log_tau, S, '.-', color=color, label=self.name)
        plt.hold(True)
        plt.fill_between(log_tau, Sl, Su, facecolor=color, alpha=0.3)
        plt.xlabel(r'log $\tau$ \ (s)')
        plt.ylabel(r'log $\langle \| x(t) - x(t-\tau) \|^2 \rangle$ (um^2)')
        if showPlot:
            plt.show()

    def plotPosturalCovariance(self, showPlot=True):
        if self.Ctheta is None:
            return
        plt.imshow(self.Ctheta, plt.get_cmap('PuOr'))
        plt.clim((-0.5, 0.5))
        plt.colorbar()
        if showPlot:
            plt.show()

    def plotPosturalModeDistribution(self, color='k', showPlot=True):
        if self.Ctheta is None:
            return
        plt.plot(np.cumsum(self.ltheta)/np.sum(self.ltheta), '.-', color=color)
        plt.xlabel('Postural Mode')
        plt.ylabel('%% Variance')
        plt.ylim((0, 1))
        if showPlot:
            plt.show()

    def plotAveragePosturalModeDistribution(self, color='k', showPlot=True):
        l, ll, lu = self.ensembleAverage(lambda x: np.cumsum(x.ltheta) / np.sum(x.ltheta))
        plt.plot(l, '.-', color=color)
        plt.hold(True)
        plt.fill_between(xrange(l.shape[0]), ll, lu,
                         facecolor=color, alpha=0.3)
        plt.xlabel('Postural Mode')
        plt.ylabel('%% Variance')
        plt.ylim((0, 1))
        if showPlot:
            plt.show()

    def plotPosturalPhaseSpaceDensity(self, postureVec1, postureVec2,
                                      showPlot=True):
        if self.Ctheta is None:
            return
        if isinstance(postureVec1, int):
            postureVec1 = self.vtheta[:, postureVec1]
        if isinstance(postureVec2, int):
                postureVec2 = self.vtheta[:, postureVec2]
        A = []
        B = []
        for traj in self:
            if traj.Ctheta is None:
                continue
            posture = traj.getMaskedPosture(traj.posture)
            missing = np.any(posture.mask, axis=1)
            a = np.dot(posture, postureVec1)
            a[missing] = ma.masked
            A.append(a)
            b = np.dot(posture, postureVec2)
            b[missing] = ma.masked
            B.append(b)
        plt.hexbin(np.concatenate(A), np.concatenate(B), bins='log')
        if showPlot:
            plt.show()


class WormTrajectoryEnsembleGroup(object):
    def __init__(self, ensembles, name=None, colorScheme=None):
        if any(not isinstance(it, WormTrajectoryEnsemble)
               for it in ensembles):
            raise TypeError('A trajectory ensemble group must contain ' +
                            'WormTrajectoryEnsemble objects.')
        self._ensembles = list(ensembles)
        self.name = name
        if colorScheme is None:
            self.colorScheme = lambda e: 'k'
        else:
            self.colorScheme = colorScheme

    def __iter__(self):
        for ens in self._ensembles:
            yield ens

    #  TODO: Implement
    #  __add__(), __radd__(), __iadd__(), __mul__(), __rmul__() and __imul__()

    def __getitem__(self, key):
        return self._ensembles[key]

    def __setitem__(self, key, value):
        self._ensembles[key] = value

    def __delitem__(self, key):
        del self._ensembles[key]

    def __contains__(self, value):
        return value in self._ensembles

    def __len__(self):
        return len(self._ensembles)

    def __getslice__(self, i, j):
        return WormTrajectoryEnsembleGroup(self._ensembles[i:j],
                                           name=self.name+' Slice')

    def __setslice__(self, i, j, sequence):
        self._ensembles[i:j] = sequence

    def __delslice__(self, i, j):
        del self._ensembles[i:j]

    def append(self, item):
        self._ensembles.append(item)

    def count(self):
        return self._ensembles.count()

    def index(self, value):
        return self._ensembles.index(value)

    def extend(self, iter):
        self._ensembles.extend(iter)

    def insert(self, index, item):
        self._ensembles.insert(index, item)

    def pop(self, index):
        self._ensembles.pop([index])

    def remove(self, item):
        self._ensembles.remove(item)

    def reverse(self):
        self._ensembles.reverse()

    def sort(self, cmp=None, key=None):
        if cmp is None and key is None:
            key = lambda e: e.name
        self._ensembles.sort(cmp=cmp, key=key)

    def calculatePosturalMeasurements(self):
        for ens in self:
            ens.calculatePosturalMeasurements()

    def tilePlots(self, plotFunc, ni=1, nj=None, showPlot=True):
        if nj is None:
            nj = np.ceil(len(self)/ni)
        if ni is None:
            ni = np.ceil(len(self)/nj)
        for i, e in enumerate(self):
            plt.subplot(ni, nj, i+1)
            plotFunc(e)
            plt.title(e.name)
        plt.show()

    def plotSpeedDistribution(self, bins=None, showPlot=True):
        for ens in self:
            color = self.colorScheme(ens)
            ens.plotSpeedDistribution(bins=bins,
                                      color=color,
                                      showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()

    def plotSpeedAutocorrelation(self, showPlot=True):
        for ens in self:
            color = self.colorScheme(ens)
            ens.plotSpeedAutocorrelation(color=color,
                                         showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()

    def plotMeanSquaredDisplacement(self, showPlot=True):
        for ens in self:
            color = self.colorScheme(ens)
            ens.plotMeanSquaredDisplacement(color=color,
                                            showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()

    def plotPosturalModeDistribution(self, showPlot=True):
        for ens in self:
            color = self.colorScheme(ens)
            ens.plotAveragePosturalModeDistribution(color=color,
                                                    showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()
