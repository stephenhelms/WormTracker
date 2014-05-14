import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import h5py

dataFile = 'D:\\2014-04-14_n2_a_b_day_7_processed.h5'

f = h5py.File(dataFile, 'r')

os.chdir('D:\\Stephen\\Documents\\Code\\wormtracker-matlab')

# Get strain list
strains = f['worms'].keys()

# Get worm ID list for each strain
wormIDs = {}

for strain in strains:
    wormIDs[strain] = [k for k, v in f['worms'][strain].items() if isinstance(v, h5py.Group)]

# Test case
worm = ('N2', '1')

def acf(x, lags=500):
    # from stackexchange
    x = x - np.mean(x)  # remove mean
    if len(lags) == 1:
        lags = range(1, lags)

    return numpy.array([1]+[numpy.corrcoef(x[:-i], x[i:])[0, 1] \
        for i in lags])


class WormTrajectory:
    def __init__(self, h5obj, strain, wormID):
        self.h5ref = h5obj['worms'][strain][wormID]
        self.frameRate = 11.5
        self.pixelsPerMicron = h5obj['video']['pixelsPerMicron'][...]
        self.foodCircle = self.h5ref['foodCircle'][...] / self.pixelsPerMicron

        self.frames = [v for v in self.h5ref.values()
                       if isinstance(v, h5py.Group)]
        self.frames.sort(cmp=lambda f1, f2: (int(f1.name.split('/')[-1]) >
                                             int(f2.name.split('/')[-1])))
        self.maxFrameNumber = max(int(f.name.split('/')[-1])
        	                      for f in self.frames) + 1
        self.t = ma.array(range(0, self.maxFrameNumber))/self.frameRate
        self.X = ma.zeros((self.maxFrameNumber,2))
        self.v = ma.zeros((self.maxFrameNumber,2))
        self.s = ma.zeros((self.maxFrameNumber,))
        self.phi = ma.empty((self.maxFrameNumber,))
        self.length = np.NaN
        self.width = np.NaN
        #self.psiEnds = np.empty((self.maxFrameNumber,2)) * np.NaN
        #self.psi = np.empty((self.maxFrameNumber,)) * np.NaN
        #self.dpsi = np.empty((self.maxFrameNumber,)) * np.NaN
        #self.state = np.empty((self.maxFrameNumber,)) * np.NaN
        #self.theta = np.empty((self.maxFrameNumber,self.numberAngles)) * np.NaN

    def identifyBadFrames(self):
        lengths = ma.zeros((self.maxFrameNumber,))
        widths = ma.zeros((self.maxFrameNumber,))
        for frame in self.frames:
            n = int(frame.name.split('/')[-1])
            lengths[n] = frame['length'][...]
            widths[n] = frame['width'][...]

        badFrames = np.logical_or(lengths.filled() == 0,
        	                      widths.filled() == 0)
        lengths.mask = badFrames
        widths.mask = badFrames
        self.length = np.median(lengths.compressed())
        self.width = np.median(widths.compressed())
        badFrames = np.logical_or(badFrames,
            np.logical_or(np.logical_or(lengths < 0.8*self.length,
                                        lengths > 1.2*self.length),
                          np.logical_or(widths < 0.5*self.width,
                                        widths > 1.5*self.width)))
        self.badFrames = badFrames.filled()

    def excludeBadFrames(self):
        self.X[self.badFrames, :] = ma.masked
        self.v[self.badFrames, :] = ma.masked
        self.s[self.badFrames] = ma.masked
        self.phi[self.badFrames] = ma.masked

    def extractCentroidMeasurements(self):
        for frame in self.frames:
            n = int(frame.name.split('/')[-1])
            self.X[n, :] = frame['centroid'][...] + frame['boundingBox'][0:2]

        self.X = self.X / self.pixelsPerMicron
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

    def plotTrajectory(self):
        plt.scatter(self.X[:, 0], self.X[:, 1])
        plt.hold(True)
        circle = plt.Circle(self.foodCircle[0:2], radius=self.foodCircle[-1],
        	                color='r', fill=False)
        plt.gca().add_patch(circle)
        plt.xlim((0, 10000))
        plt.ylim((0, 10000))
        plt.xlabel('x (um)')
        plt.xlabel('y (um)')
        plt.show()

    def plotSpeed(self):
        plt.plot(self.t, self.s, 'k.')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (um/s)')
        plt.show()

    def plotBearing(self):
        plt.plot(self.t, self.phi/np.pi, 'k.')
        plt.xlabel('Time (s)')
        plt.ylabel('Bearing ($\pi$ rad)')
        plt.show()

    def plotSpeedDistribution(self, bins=None):
        if bins is None:
            bins = np.ceil(np.sqrt(np.sum(np.logical_not(self.badFrames))))

        plt.hist(self.s.compressed(), bins, normed=True)
        plt.xlabel('Speed (um/s)')
        plt.ylabel('Probability')
        plt.show()

    def plotSpeedAutocorrelation(self, maxT=100):
    	n = np.round(maxT*self.frameRate)
    	tau = range(n)/self.frameRate
    	acf = acf(self.s, n)
    	plt.semilogx(tau, acf, 'k-')
    	xlabel(r'$\log \tau / (s)$')
    	ylabel(r'$\langle s(t) \cdot s(t+\tau)\rangle$')

    def plotBearingAutocorrelation(self, maxT=100):
    	n = np.round(maxT*self.frameRate)
    	tau = range(n)/self.frameRate


    def plotMeanSquaredDisplacement(self):
        tau = np.logspace(-1,3,200)
        lags = np.round(tau*self.frameRate)
        Sigma = ma.zeros((200,))
        for i, lag in enumerate(lags):
            displacements = self.X[lag:, :] - self.X[:-lag, :]
            Sigma[i] = np.mean(np.log10(np.sum(displacements**2, axis=1)))

        plt.plot(np.log10(tau), Sigma, 'k.')
        plt.xlabel(r'log $\tau$ \ (s)')
        plt.ylabel(r'log $\langle \| x(t) - x(t-\tau) \|^2 \rangle$ (um^2)')
        plt.show()

        return (tau,Sigma)
