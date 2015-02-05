import os
import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import h5py
import itertools
import collections
import cv2
import wormtracker.wormimageprocessor as wip
import multiprocessing as multi
#from numba import jit
import scipy.stats as ss
from tsstats import *
from stats import *
from copy import deepcopy


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


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def meanSquaredDisplacement(X, lags=500, exclude=None):
    if exclude is None:
        exclude = np.zeros(X.shape[0])
    exclude = np.cumsum(exclude.astype(int))

    if type(lags) is int:
        lags = xrange(1,lags)

    Sigma = ma.zeros((len(lags),))
    for i, lag in enumerate(lags):
        x0 = X[lag:, :].copy()
        x1 = X[:-lag, :].copy()
        reject = (exclude[lag:]-exclude[:-lag])>0
        x0[reject, :] = ma.masked
        x1[reject, :] = ma.masked
        displacements = x0 - x1
        d2 = (displacements**2).sum(axis=1).compressed()
        ld2 = np.log10(d2)
        ld2[np.isinf(ld2)] = ma.masked
        Sigma[i] = np.mean(ld2)
    return Sigma


class WormTrajectory:
    def __init__(self, h5obj, strain, wormID, videoFilePath=None, frameRange=None):
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
        self.length = self.h5ref['avgLength'][0]
        self.width = self.h5ref['avgWidth'][0]
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

        self.revBoundaries = None
        self.nearRev = np.zeros(self.t.shape, 'bool')
        self.excluded = np.zeros(self.t.shape, 'bool')
        self.state = None

        if frameRange is not None:
            self.frameRange = frameRange
            self.clearAnalysisVariables()
            self.isolateFrameRange(frameRange)
        else:
            self.frameRange = None

    def __deepcopy__(self, memo):
        traj = WormTrajectory(self.h5obj, deepcopy(self.strain, memo),
                              deepcopy(self.wormID, memo),
                              frameRange=deepcopy(self.frameRange, memo))
        if self.videoFile is not None:
            traj.videoFile = deepcopy(self.videoFile, memo)
        if self.excluded is not None:
            traj.excluded = deepcopy(self.excluded, memo)
        return traj

    def clearAnalysisVariables(self):
        self.Ctheta = None
        self.ltheta = None
        self.vtheta = None

        self.revBoundaries = None
        self.nearRev = np.zeros(self.t.shape, 'bool')
        self.state = None

    def isolateTimeRange(self, timeRange):
        self.isolateFrameRange(np.round(np.array(timeRange)*self.frameRate).astype(int))

    def isolateFrameRange(self, frameRange):
        if frameRange[0] < 0:
            raise IndexError('Negative frame.')
        if frameRange[1] > self.t.shape:
            frameRange[1] = self.t.shape[0]
        self.frameRange = frameRange

        self.t = self.t[frameRange[0]:frameRange[1]]
        self.X = self.X[frameRange[0]:frameRange[1], :]
        self.Xhead = self.Xhead[frameRange[0]:frameRange[1], :]
        self.v = self.v[frameRange[0]:frameRange[1], :]
        self.s = self.s[frameRange[0]:frameRange[1]]
        self.phi = self.phi[frameRange[0]:frameRange[1]]
        self.psi = self.psi[frameRange[0]:frameRange[1]]
        self.dpsi = self.dpsi[frameRange[0]:frameRange[1]]
        self.badFrames = self.badFrames[frameRange[0]:frameRange[1]]
        self.allCentroidMissing = np.all(self.badFrames)
        self.skeleton = self.skeleton[frameRange[0]:frameRange[1], ...]
        self.posture = self.posture[frameRange[0]:frameRange[1], :]
        self.orientationFixed = self.orientationFixed[frameRange[0]:frameRange[1]]
        self.allPostureMissing = np.all(np.logical_not(self.orientationFixed))
        self.excluded = self.excluded[frameRange[0]:frameRange[1]]

        self.clearAnalysisVariables()

    def asWindows(self, windowSize=100., overlap=0.5):
        nWindows = int(np.round((self.t[-1] - self.t[0]) / windowSize / (1.-overlap)))
        for i in xrange(nWindows):
            tRange = self.t[0] + i*windowSize*(1.-overlap) + (0, windowSize)
            traj = deepcopy(self)
            traj.isolateTimeRange(tRange)
            yield traj

    def identifyReversals(self, transitionWindow=2.):
        dpsi = self.getMaskedPosture(self.dpsi)
        rev = np.abs(dpsi)>np.pi/2.

        ii = 1
        inRev = False
        state = np.zeros(self.t.shape, int)
        state[~rev] = 1
        state[rev] = 2
        if np.any(rev.mask):
            state[rev.mask] = 0
        self.state = state

        revBoundaries = ma.empty((1,2),int)
        for j in xrange(1,self.t.shape[0]-2):
            if not inRev:
                if (state[j]==2 & state[j+1]==2 |
                    state[j]==2 & state[j+1]==0 & state[j+2]==2):
                    if state[j-1] == 0:
                        revBoundaries[ii,0] = ma.masked
                    else:
                        revBoundaries[ii,0] = j
                    inRev = True
            else:
                if (state[j]==1 & state[j+1]==1 |
                    state[j]==1 & state[j+1]==0 & state[j+2]==1):
                    revBoundaries[ii,1] = j
                    inRev = False
                    ii = ii+1
                elif (state[j]==1 & state[j+1]==0 & state[j+2]==0):
                    revBoundaries[ii,1] = j
                    inRev = False
                    ii = ii+1
                elif (state[j]==0 & state[j+1]==0):
                    revBoundaries[ii,1] = ma.masked
                    inRev=False
                    ii = ii+1

        if revBoundaries[-1,1] == 0:
            revBoundaries[-1,1] = ma.masked
        self.revBoundaries = revBoundaries

        revEdges = self.revBoundaries[:].compressed()
        for boundary in revEdges:
            self.nearRev[(self.t>boundary-transitionWindow/2.) &
                         (self.t<boundary+transitionWindow/2.)] = True


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

    def calculatePosturalMeasurements(self):
        posture = self.getMaskedPosture(self.posture)
        missing = np.any(posture.mask, axis=1)
        if np.all(missing):
            self.Ctheta = None
            self.ltheta = None
            self.vtheta = None
        else:
            posture = posture[~missing, :].T
            self.Ctheta = np.cov(posture)
            self.ltheta, self.vtheta = LA.eig(self.Ctheta)

    def getMaskedCentroid(self, data):
        data = ma.array(data)
        sel = self.badFrames | self.excluded
        data[sel, ...] = ma.masked
        data[np.isnan(data)] = ma.masked
        return data

    def getMaskedPosture(self, data):
        data = ma.array(data)
        sel = np.logical_or(np.logical_not(self.orientationFixed),
                            self.badFrames) | self.excluded
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
        plt.plot(X[:, 0], X[:, 1], '.', color=color)
        plt.hold(True)
        if self.foodCircle is not None:
            circle = plt.Circle(self.foodCircle[0:2],
                                radius=self.foodCircle[-1],
                                color='r', fill=False)
            plt.gca().add_patch(circle)
        plt.xlim((0, 10000))
        plt.ylim((0, 10000))
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        plt.gca().set_aspect('equal')
        if showPlot:
            plt.show()

    def plotTrajectoryVectors(self, showFrame=True, showPlot=True):
        if showFrame and self.firstFrame is not None:
            plt.imshow(self.firstFrame, plt.gray(),
                       origin='lower',
                       extent=(0,
                               self.firstFrame.shape[1]/self.pixelsPerMicron,
                               0,
                               self.firstFrame.shape[0]/self.pixelsPerMicron))
            plt.hold(True)
        X = self.getMaskedCentroid(self.X)
        phi = self.getMaskedCentroid(self.phi)
        psi = self.getMaskedPosture(self.psi)
        s = self.getMaskedCentroid(self.s)
        plt.quiver(X[:,0], X[:,1], (s+10.)*np.cos(phi), (s+10.)*np.sin(phi),
                   color='k', units='xy')
        psi = self.getMaskedPosture(self.psi)
        mu = s.mean()
        plt.quiver(X[:,0], X[:,1], mu*np.cos(psi), mu*np.sin(psi),
                   color='r', units='xy')
        if self.foodCircle is not None:
            circle = plt.Circle(self.foodCircle[0:2],
                                radius=self.foodCircle[-1],
                                color='r', fill=False)
            plt.gca().add_patch(circle)
        plt.xlim((0, 10000))
        plt.ylim((0, 10000))
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
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

    def plotBodyBearing(self, showPlot=True):
        psi = self.getMaskedPosture(self.psi)
        plt.plot(self.t, psi/np.pi, 'k.')
        plt.xlabel('Time (s)')
        plt.ylabel('Body Bearing ($\pi$ rad)')
        if showPlot:
            plt.show()

    def getSpeedDistribution(self, bins, useKernelDensity=True,
                             ignoreReversalTransition=True):
        s = self.getMaskedCentroid(self.s)
        if ignoreReversalTransition:
            if self.revBoundaries is None:
                self.identifyReversals()
            s[self.nearRev] = ma.masked
        if not useKernelDensity:
            out = np.histogram(s.compressed(), bins,
                               density=True)[0]
        else:
            kd = ss.gaussian_kde(s.compressed())
            out = kd.evaluate(bins)
        return out

    def plotSpeedDistribution(self, bins=None, color='k', useKernelDensity=True,
                              ignoreReversalTransition=True, showPlot=True):
        if not useKernelDensity:
            if bins is None:
                bins = np.ceil(np.sqrt(np.sum(np.logical_not(self.badFrames))))
            s = self.getMaskedCentroid(self.s)
            if self.revBoundaries is None:
                self.identifyReversals()
            s[self.nearRev] = ma.masked
            plt.hist(s.compressed(), bins, normed=True, facecolor=color)
        else:
            if bins is None:
                bins = np.linspace(0,500,500)
            D = self.getSpeedDistribution(bins,
                    ignoreReversalTransition=ignoreReversalTransition)
            plt.plot(bins, D, 'k-')
        plt.xlabel('Speed (um/s)')
        plt.ylabel('Probability Density')
        if showPlot:
            plt.show()

    def getSpeedAutocorrelation(self, maxT=10., windowSize=100.):
        lags = np.arange(0, np.round(maxT*self.frameRate))
        if windowSize is None:
            s = self.getMaskedCentroid(self.s)
            s[self.nearRev] = ma.masked
            C = s.var()*acf(s, lags)
        else:
            def result(traj):
                traj.identifyReversals()
                s = traj.getMaskedCentroid(traj.s)
                s[traj.nearRev] = ma.masked
                return s.var()*acf(s, lags)

            C = np.array([result(traj)
                          for traj in self.asWindows(windowSize)]).T
            C = C.mean(axis=1)
        tau = lags / self.frameRate

        return tau, C

    def plotSpeedAutocorrelation(self, maxT=10., windowSize=100.,
                                 color='k', showPlot=True):
        tau, C = self.getSpeedAutocorrelation(maxT, windowSize)
        plt.plot(tau, C, '-', color=color)
        plt.xlabel(r'$\tau$ (s)')
        plt.ylabel(r'$\langle \hat{s}(0) \cdot \hat{s}(\tau)\rangle \mathrm{(um/s)}^2$')
        if showPlot:
            plt.show()

    def getBodyBearingAutocorrelation(self, maxT=50., windowSize=100.):
        lags = np.round(np.linspace(0, np.round(maxT*self.frameRate), 200)).astype(int)
        if windowSize is None:
            psi = self.getMaskedPosture(self.psi)
            C = dotacf(ma.array([np.cos(psi),np.sin(psi)]).T, lags)
        else:
            def result(traj):
                psi = traj.getMaskedPosture(traj.psi)
                return dotacf(ma.array([np.cos(psi),np.sin(psi)]).T, lags)

            C = np.array([result(traj)
                          for traj in self.asWindows(windowSize)]).T
            C = C.mean(axis=1)
        tau = lags / self.frameRate
        return tau, C

    def plotBodyBearingAutocorrelation(self, maxT=50., windowSize=100.,
                                       color='k', showPlot=True):
        tau, C = self.getBodyBearingAutocorrelation(maxT, windowSize=windowSize)
        plt.plot(tau, C, '-', color=color)
        plt.xlabel(r'$\tau \mathrm{(s)}$')
        plt.ylabel(r'$\langle \vec{\psi}(0) \cdot \vec{\psi}(\tau) \rangle$')
        if showPlot:
            plt.show()

    def getReversalStateAutocorrelation(self, maxT=100):
        n = int(np.round(maxT*self.frameRate))
        tau = range(n)/self.frameRate
        dpsi = self.getMaskedPosture(self.dpsi)
        C = acf(np.cos(dpsi), n)
        return tau, C

    def plotReversalStateAutocorrelation(self, maxT=100, color='k', showPlot=True):
        tau, C = self.getReversalStateAutocorrelation(maxT)
        plt.plot(tau, C, '-', color=color)
        plt.xlabel(r'$\tau (s)$')
        plt.ylabel(r'$\langle \cos \Delta\psi(t) \cdot \cos \Delta\psi(t+\tau) \rangle$')
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
        plt.plot(tau, C, '-', color=color)
        plt.xlabel(r'$\tau / (s)$')
        plt.ylabel(r'$\langle \cos\left[\psi(t)-\psi(t+\tau)\right]\rangle$')
        if showPlot:
            plt.show()

    def getMeanSquaredDisplacement(self, tau=None):
        if tau is None:
            tau = np.logspace(-1, 2, 200)

        lags = np.round(tau*self.frameRate)
        X = self.getMaskedCentroid(self.X)
        Sigma = meanSquaredDisplacement(X, lags, self.excluded)

        return (tau, Sigma)

    def plotMeanSquaredDisplacement(self, tau=None, Dworm=100., showPlot=True,
                                    showRef=True):
        tau, Sigma = self.getMeanSquaredDisplacement(tau)
        plt.plot(np.log10(tau), Sigma, 'k.')
        if showRef:
            s = self.getMaskedCentroid(self.s)
            plt.plot(np.log10(tau), np.log10((s.mean()**2)*(tau**2)), 'r-')
            alpha = 4.*Dworm
            plt.plot(np.log10(tau), np.log10(alpha*tau), 'r:')
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
        plt.ylabel(r'% Variance')
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
        plt.xlabel('Time (s)')
        plt.ylabel('Projection')
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
        posture = self.getMaskedPosture(self.posture)
        missing = np.any(posture.mask, axis=1)
        A = np.dot(posture, postureVec1)
        A[missing] = ma.masked
        B = np.dot(posture, postureVec2)
        B[missing] = ma.masked
        xmin = A.min()
        xmax = A.max()
        ymin = B.min()
        ymax = B.max()
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([A.compressed(), B.compressed()])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        plt.imshow(np.rot90(Z), extent=[xmin,xmax,ymin,ymax])
        plt.show()
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

    def plotPosturalPhaseSpace3D(self, postureVec1, postureVec2, postureVec3,
                                 nMaxPts=1000, color='k', showPlot=True):
        if self.Ctheta is None:
            return
        if isinstance(postureVec1, int):
            postureVec1 = self.vtheta[:, postureVec1]
        if isinstance(postureVec2, int):
            postureVec2 = self.vtheta[:, postureVec2]
        if isinstance(postureVec3, int):
            postureVec3 = self.vtheta[:, postureVec3]
        posture = self.getMaskedPosture(self.posture)
        missing = np.any(posture.mask, axis=1)
        A = np.dot(posture, postureVec1)
        A[missing] = ma.masked
        B = np.dot(posture, postureVec2)
        B[missing] = ma.masked
        C = np.dot(posture, postureVec3)
        C[missing] = ma.masked
        A = A.compressed()
        B = B.compressed()
        C = C.compressed()
        if A.shape[0] > nMaxPts:
            plotPts = np.random.choice(A.shape[0], nMaxPts, replace=False)
            A = A[plotPts]
            B = B[plotPts]
            C = C[plotPts]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(A,B,C,
                   marker='o', c='k', s=5)
        ax.set_xlabel('Mode 1')
        ax.set_ylabel('Mode 2')
        ax.set_zlabel('Mode 3')
        if showPlot:
            plt.show()


class WormTrajectoryEnsemble:
    def __init__(self, trajectoryIter=None, name=None, nameFunc=None, color='k'):
        if any(not isinstance(it, WormTrajectory) for it in trajectoryIter):
            raise TypeError('A trajectory ensemble must contain ' +
                            'WormTrajectory objects.')
        self._trajectories = list(trajectoryIter)
        self.name = name
        if nameFunc is None:
            nameFunc = lambda t: t.strain + ' ' + t.wormID
        self.nameFunc = nameFunc
        self.color = color

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

    def plotSpeedDistribution(self, bins=None, color=None, showPlot=True):
        if bins is None:
            bins = np.linspace(0, 500, 200)
        if color is None:
            color = self.color
        p, pl, pu = self.ensembleAverage(lambda x: x.getSpeedDistribution(bins))
        plt.plot(bins, p, '.-', color=color, label=self.name)
        plt.hold(True)
        plt.fill_between(bins, pl, pu, facecolor=color, alpha=0.3)
        plt.xlabel('Speed (um/s)')
        plt.ylabel('Probability')
        plt.xlim([0,max(bins)])
        plt.grid(True)
        if showPlot:
            plt.show()

    def plotSpeedDistributions(self, bins=None, showPlot=True):
        if bins is None:
            bins = np.linspace(0, 500, 200)
        for i, t in enumerate(self):
            color = plt.cm.jet(float(i)/float(len(self)-1))
            p = t.getSpeedDistribution(bins)[0]
            plt.plot(bins,p,'.-',color=color,
                     label=self.nameFunc(t))
            plt.hold(True)
        plt.xlabel('Speed (um/s)')
        plt.ylabel('Probability')
        plt.xlim([0, max(bins)])
        plt.grid(True)
        plt.legend()
        if showPlot:
            plt.show()

    def plotSpeedAutocorrelation(self, maxT=100, color=None, showPlot=True):
        if color is None:
            color = self.color
        # assume all same frame rate
        n = int(np.round(maxT*self[0].frameRate))
        tau = range(n)/self[0].frameRate
        C, Cl, Cu = self.ensembleAverage(lambda x: x.getSpeedAutocorrelation(maxT)[1])
        plt.plot(tau, C, '.-', color=color, label=self.name)
        plt.fill_between(tau, Cl, Cu, facecolor=color, alpha=0.3)
        plt.xlabel(r'$\tau (s)$')
        plt.ylabel(r'$\langle s(t) \cdot s(t+\tau)\rangle$')
        plt.grid(True)
        # TODO: smart xlim
        if showPlot:
            plt.show()

    def plotBearingAutocorrelation(self, maxT=100, color=None, showPlot=True):
        if color is None:
            color = self.color
        # assume all same frame rate
        n = int(np.round(maxT*self[0].frameRate))
        tau = range(n)/self[0].frameRate
        C, Cl, Cu = self.ensembleAverage(lambda x: x.getBearingAutocorrelation(maxT)[1])
        plt.plot(tau, C, '.-', color=color, label=self.name)
        plt.fill_between(tau, Cl, Cu, facecolor=color, alpha=0.3)
        plt.xlabel(r'$\tau / (s)$')
        plt.ylabel(r'$\langle \vec{\psi}(0) \cdot \vec{\psi}(\tau)\rangle$')
        plt.grid(True)
        # TODO: smart xlim
        if showPlot:
            plt.show()

    def plotMeanSquaredDisplacement(self, tau=None, color=None, showPlot=True):
        if color is None:
            color = self.color
        tau = np.logspace(-1,3,200)
        S, Sl, Su = self.ensembleAverage(lambda x: x.getMeanSquaredDisplacement(tau)[1])
        log_tau = np.log10(tau)
        plt.plot(log_tau, S, '.-', color=color, label=self.name)
        plt.hold(True)
        plt.fill_between(log_tau, Sl, Su, facecolor=color, alpha=0.3)
        plt.xlabel(r'log $\tau$ \ (s)')
        plt.ylabel(r'log $\langle \| x(t) - x(t-\tau) \|^2 \rangle$ (um^2)')
        plt.xlim((np.min(np.log10(tau)), np.max(np.log10(tau))))
        plt.grid(True)
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

    def plotPosturalModeDistribution(self, color=None, showPlot=True):
        if self.Ctheta is None:
            return
        if color is None:
            color = self.color
        plt.plot(np.cumsum(self.ltheta)/np.sum(self.ltheta), '.-', color=color)
        plt.xlabel('Postural Mode')
        plt.ylabel('%% Variance')
        plt.ylim((0, 1))
        if showPlot:
            plt.show()

    def plotAveragePosturalModeDistribution(self, color=None, showPlot=True):
        if color is None:
            color = self.color
        l, ll, lu = self.ensembleAverage(lambda x: np.cumsum(x.ltheta) / np.sum(x.ltheta))
        plt.plot(l, '.-', color=color, label=self.name)
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
    def __init__(self, ensembles, name=None):
        if any(not isinstance(it, WormTrajectoryEnsemble)
               for it in ensembles):
            raise TypeError('A trajectory ensemble group must contain ' +
                            'WormTrajectoryEnsemble objects.')
        self._ensembles = list(ensembles)
        self.name = name
        
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
                                           name=self.name)

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
            ens.plotSpeedDistribution(bins=bins,
                                      showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()

    def plotSpeedAutocorrelation(self, showPlot=True):
        for ens in self:
            ens.plotSpeedAutocorrelation(showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()

    def plotMeanSquaredDisplacement(self, tau=None, showPlot=True):
        if tau is None:
            tau = np.logspace(-1,3,80)
        for ens in self:
            ens.plotMeanSquaredDisplacement(tau=tau, showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()

    def plotPosturalModeDistribution(self, showPlot=True):
        for ens in self:
            ens.plotAveragePosturalModeDistribution(showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()
