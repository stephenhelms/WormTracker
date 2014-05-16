import os
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import itertools
import cv2
import wormimageprocessor as wip


def configureMatplotLibStyle():
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.labelsize'] = 'x-large'
    mpl.rcParams['axes.titlesize'] = 'x-large'
    mpl.rcParams['grid.color'] = (0.5, 0.5, 0.5)
    mpl.rcParams['legend.fontsize'] = 'medium'
    mpl.rcParams['legend.numpoints'] = 1
    mpl.rcParams['legend.frameon'] = False


def acf(x, lags=500):
    # from stackexchange
    x = x - np.mean(x)  # remove mean
    if type(lags) is int:
        lags = range(1, lags)

    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0, 1] \
        for i in lags])


def dotacf(x, lags=500):
    if type(lags) is int:
        lags = xrange(lags)
    return [np.mean(np.dot(x[l:, :], x[:l, :]), axis=0)
            for l in lags]


class WormTrajectory:
    filterByWidth = False
    firstFrame = None
    attr = {}

    def __init__(self, h5obj, strain, wormID, videoFilePath=None):
        self.h5obj = h5obj
        self.h5ref = h5obj['worms'][strain][wormID]
        self.strain = strain
        self.wormID = wormID
        self.frameRate = h5obj['/video/frameRate'][...]
        self.pixelsPerMicron = h5obj['/video/pixelsPerMicron'][...]
        self.foodCircle = self.h5ref['foodCircle'][...] / self.pixelsPerMicron
        self.t = self.h5ref['time'][...]
        self.maxFrameNumber = self.t.shape[0]
        self.X = ma.array(self.h5ref['centroid'][...] / self.pixelsPerMicron)
        self.v = ma.zeros(self.X.shape)
        self.s = ma.zeros((self.maxFrameNumber,))
        self.phi = ma.zeros((self.maxFrameNumber,))
        self.length = np.NaN
        self.width = np.NaN
        self.badFrames = np.zeros((self.maxFrameNumber,), dtype='bool')
        if videoFilePath is not None:
            videoFile = h5obj['/video/videoFile'][0]
            self.videoFile = os.path.join(videoFilePath, videoFile)
        else:
            self.videoFile = None

        #self.psiEnds = np.empty((self.maxFrameNumber,2)) * np.NaN
        #self.psi = np.empty((self.maxFrameNumber,)) * np.NaN
        #self.dpsi = np.empty((self.maxFrameNumber,)) * np.NaN
        #self.state = np.empty((self.maxFrameNumber,)) * np.NaN
        #self.theta = np.empty((self.maxFrameNumber,self.numberAngles)) * np.NaN

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

    def identifyBadFrames(self):
        lengths = self.h5ref['length'][...]
        widths = self.h5ref['width'][...]
        
        badFrames = np.logical_or(lengths == 0,
        	                      widths == 0)
        
        self.length = np.median(lengths[np.logical_not(badFrames)])
        self.width = np.median(widths[np.logical_not(badFrames)])
        if self.filterByWidth:
            badFrames = np.logical_or(badFrames,
                np.logical_or(np.logical_or(lengths < 0.8*self.length,
                                            lengths > 1.2*self.length),
                              np.logical_or(widths < 0.5*self.width,
                                            widths > 1.5*self.width)))
        else:
            badFrames = np.logical_or(badFrames,
                np.logical_or(lengths < 0.8*self.length,
                              lengths > 1.2*self.length))
        self.badFrames = badFrames

    def excludeBadFrames(self):
        self.X[self.badFrames, :] = ma.masked
        self.v[self.badFrames, :] = ma.masked
        self.s[self.badFrames] = ma.masked
        self.phi[self.badFrames] = ma.masked

    def extractCentroidMeasurements(self):
        self.X[self.badFrames, :] = ma.masked
        self.v[1:-1] = (self.X[2:, :] - self.X[0:-2])/(2.0/self.frameRate)
        self.s = np.sqrt(np.sum(np.power(self.v, 2), axis=1))
        self.phi = np.arctan2(self.v[:, 1], self.v[:, 0])
        self.excludeBadFrames()

    def extractPosturalData(self):
        # import skeleton splines
        skeletons = [None] * self.maxFrameNumber
        for frame in self.frames:
            n = int(frame.name.split('/')[-1])
            # check bad frame
            if frame['length'] > 0 and not self.badFrames[n]:
                skeletons[n] = frame['skeletonSpline'][...]

        self.haveSkeleton = [(skeleton is not None) for skeleton in skeletons]

    def fixPosturalOrdering(self, skeletons):
        # compare possible skeleton orientations
        interframe_d = np.empty((self.maxFrameNumber, 2)) * np.NaN
        flipped = np.zeros((self.maxFrameNumber,), dtype=bool)
        nFromLastGood = np.empty((self.maxFrameNumber,)) * np.NaN

        def skeletonDist(skeleton1, skeleton2):
            distEachPoint = np.sqrt(np.sum(np.power(skeleton1 -
                                                    skeleton2, 2),
                                           axis=1))
            # return average distance per spline point
            return np.sum(distEachPoint)/skeleton1.shape[0]

        for i in xrange(1, self.maxFrameNumber):
            # check whether there is a previous skeleton to compare
            if not self.haveSkeleton[i] or not np.any(self.haveSkeleton[:i]):
                continue

            ip = np.where(self.haveSkeleton[:i])[0][-1]  # last skeleton
            nFromLastGood[i] = i - ip
            interframe_d[i, 0] = skeletonDist(skeletons[i], skeletons[ip])
            # flipped orientation
            interframe_d[i, 1] = skeletonDist(np.flipud(skeletons[i]),
                                              skeletons[ip])
            if interframe_d[i, 1] < interframe_d[i, 0]:
                # if the flipped orientation is better, flip the data
                flipped[i] = not flipped[ip]
            else:
                flipped[i] = flipped[ip]
        
        # flip data appropriately
        # this code needs to be tested
        nAngles = max(len(skeleton) for skeleton in skeletons
        	          if skeleton is not None)
        self.skeleton = ma.zeros((self.maxFrameNumber, nAngles, 2))
        sel = self.haveSkeleton and not flipped
        self.skeleton[sel, :, :] = np.array(skeletons[sel])
        sel = self.haveSkeleton and flipped
        self.skeleton[sel, :, :] = np.array(np.flipud(skeletons[sel]))
        self.skeleton[not self.haveSkeleton, :, :] = ma.masked

        self.theta = ma.zeros((self.maxFrameNumber, nAngles))
        for frame in self.frames:
            n = int(frame.name.split('/')[-1])
            # check bad frame
            if self.haveSkeleton[n]:
            	if not flipped[n]:
                    self.theta[n,:] = frame['theta'][...]
                else:
                	self.theta[n,:] = np.flipud(frame['theta'][...])
        self.theta[not self.haveSkeleton, :] = ma.masked

    def segment(self):
        # break video into segments with matched skeletons
        max_n_missing = 10
        max_d = 10/self.pixelsPerMicron
        max_segment_frames = 500
        min_segment_size = 150

        ii = 0
        segments = []
        while ii < self.maxFrameNumber:
            begin = ii
            ii+=1
            # Continue segment until >max_n_missing consecutive bad frames
            # are found, or >max_segment_frames are collected
            n_missing = 0
            last_missing = False
            while (ii < self.maxFrameNumber and
                   ii - begin < max_segment_frames and
                   (interframe_d[ii, 0] == np.NaN or
                   	np.min(interframe_d[ii, :])<max_d)):
                if not haveSkeleton[ii]:
                	n_missing+=1
                	last_missing = True
                	if n_missing > max_n_missing:
                		ii+=1
                		break
                else:
                	n_missing = 0
                	last_missing = False
                ii+=1
            segments.append([begin, ii])

        self.segments = [segment for segment in segments
                         if np.sum(haveSkeleton[segment[0]:segment[1]]) >
                         min_segment_size]

    def assignHeadTail(self):
        # add head assignment algorithm
        raise NotImplemented()

    def plotTrajectory(self, color='k', showFrame=True, showPlot=True):
        if showFrame and self.firstFrame is not None:
            plt.imshow(self.firstFrame, plt.gray(),
                       origin='lower',
                       extent=(0,
                               self.firstFrame.shape[1]/self.pixelsPerMicron,
                               0,
                               self.firstFrame.shape[0]/self.pixelsPerMicron))
            plt.hold(True)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=color, s=10)
        plt.hold(True)
        circle = plt.Circle(self.foodCircle[0:2], radius=self.foodCircle[-1],
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
        plt.plot(self.t, self.s, 'k.')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (um/s)')
        if showPlot:
            plt.show()

    def plotBearing(self, showPlot=True):
        plt.plot(self.t, self.phi/np.pi, 'k.')
        plt.xlabel('Time (s)')
        plt.ylabel('Bearing ($\pi$ rad)')
        if showPlot:
            plt.show()

    def getSpeedDistribution(self, bins=None):
        if bins is None:
            bins = np.ceil(np.sqrt(np.sum(np.logical_not(self.badFrames))))

        out = np.histogram(self.s.compressed(), bins,
                           density=True)
        return out

    def plotSpeedDistribution(self, bins=None, color='k', showPlot=True):
        if bins is None:
            bins = np.ceil(np.sqrt(np.sum(np.logical_not(self.badFrames))))

        plt.hist(self.s.compressed(), bins, normed=True, facecolor=color)
        plt.xlabel('Speed (um/s)')
        plt.ylabel('Probability')
        if showPlot:
            plt.show()

    def getSpeedAutocorrelation(self, maxT=100):
        n = np.round(maxT*self.frameRate)
        tau = range(n)/self.frameRate
        C = acf(self.s, n)
        return tau, C

    def plotSpeedAutocorrelation(self, maxT=100):
    	tau, C = self.getSpeedAutocorrelation(maxT)
        plt.semilogx(tau, C, 'k-')
    	plt.xlabel(r'$\log \tau / (s)$')
    	plt.ylabel(r'$\langle s(t) \cdot s(t+\tau)\rangle$')

    def plotBearingAutocorrelation(self, maxT=100):
    	n = np.round(maxT*self.frameRate)
    	tau = range(n)/self.frameRate

    def getMeanSquaredDisplacement(self, tau=None):
        if tau is None:
            tau = np.logspace(-1,3,200)

        lags = np.round(tau*self.frameRate)
        Sigma = ma.zeros((200,))
        for i, lag in enumerate(lags):
            displacements = self.X[lag:, :] - self.X[:-lag, :]
            Sigma[i] = np.mean(np.log10(np.sum(displacements**2, axis=1)))

        return (tau, Sigma)

    def plotMeanSquaredDisplacement(self, tau=None):
        tau, Sigma = self.getMeanSquaredDisplacement(tau)
        plt.plot(np.log10(tau), Sigma, 'k.')
        plt.xlabel(r'log $\tau$ \ (s)')
        plt.ylabel(r'log $\langle \| x(t) - x(t-\tau) \|^2 \rangle$ (um^2)')
        plt.show()


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

    def __getitem__(self, key):
        return self._trajectories[key]

    def __setitem__(self, key, value):
        self._trajectories[key] = value

    def __delitem__(self, key):
        del self._trajectories

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

    def sort(self, cmp=None, key=None):
        if cmp is None and key is None:
            key = lambda t: int(t.wormID)
        self._trajectories.sort(cmp=cmp, key=key)

    def processAll(self):
        for t in self:
            t.readFirstFrame()
            t.identifyBadFrames()
            t.extractCentroidMeasurements()

    def ensembleAverage(self, compFunc, nSamples=1000):
        samples = np.array([compFunc(traj) for traj in self])
        return bootstrap(samples, nSamples)

    def tilePlots(self, plotFunc):
        plt.figure()
        for i, t in enumerate(self):
            plt.subplot(4, np.ceil(len(self)/4), i+1)
            plotFunc(t)
            plt.title(self.nameFunc(t))
        plt.show()

    def plotSpeedDistribution(self, color='k', showPlot=True):
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

    def plotSpeedDistributions(self, showPlot=True):
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
        n = np.round(maxT*self[0].frameRate)
        tau = range(n)/self[0].frameRate
        C, Cl, Cu = self.ensembleAverage(lambda x: x.getSpeedAutocorrelation(maxT)[1])
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


class WormTrajectoryEnsembleGroup(object):
    def __init__(self, ensembles, name=None, colorScheme=None):
        if any(not isinstance(it, WormTrajectoryEnsemble)
               for it in ensembles):
            raise TypeError('A trajectory ensemble group must contain ' +
                            'WormTrajectoryEnsemble objects.')
        self._ensembles = list(ensembles)
        self.name = name
        if colorScheme is None:
            self.colorScheme = {ens: 'k' for ens in ensembles}
        else:
            self.colorScheme = colorScheme

    def __iter__(self):
        for ens in self._ensembles:
            yield ens

    def __getitem__(self, key):
        return self._ensembles[key]

    def __setitem__(self, key, value):
        self._ensembles[key] = value

    def __delitem__(self, key):
        del self._ensembles

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

    def sort(self, cmp=None, key=None):
        if cmp is None and key is None:
            key = lambda e: e.name
        self._ensembles.sort(cmp=cmp, key=key)

    def plotSpeedDistribution(self, showPlot=True):
        plt.figure()
        for ens in self:
            color = (self.colorScheme[ens]
                     if ens in self.colorScheme
                     else 'k')
            ens.plotSpeedDistribution(color=color,
                                      showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()

    def plotSpeedAutocorrelation(self, showPlot=True):
        plt.figure()
        for ens in self:
            color = (self.colorScheme[ens]
                     if ens in self.colorScheme
                     else 'k')
            ens.plotSpeedAutocorrelation(color=color,
                                         showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()

    def plotMeanSquaredDisplacement(self, showPlot=True):
        plt.figure()
        for ens in self:
            color = (self.colorScheme[ens]
                     if ens in self.colorScheme
                     else 'k')
            ens.plotMeanSquaredDisplacement(color=color,
                                            showPlot=False)
        plt.legend()
        if showPlot:
            plt.show()
