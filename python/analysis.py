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

        badFrames = np.logical_or(lengths == 0, widths == 0)
        self.length = np.median(ma.array(lengths, mask=badFrames))
        self.width = np.median(ma.array(widths, mask=badFrames))
        badFrames = np.logical_or(badFrames,
            np.logical_or(np.logical_or(lengths < 0.8*self.length,
                                        lengths > 1.2*self.length),
                          np.logical_or(widths < 0.5*self.width,
                                        widths > 1.5*self.width)))
        self.badFrames = badFrames

    def excludeBadFrames(self):
        self.X[self.badFrames, :] = ma.masked
        self.v[self.badFrames, :] = ma.masked
        self.s[self.badFrames] = ma.masked
        self.phi[self.badFrames] = ma.masked

    def extractCentroidMeasurements(self):
        for frame in self.frames:
            n = int(frame.name.split('/')[-1])
            self.X[n, :] = frame['centroid'][...]  + frame['boundingBox'][0:2]

        self.X = self.X / self.pixelsPerMicron
        self.X[self.badFrames, :] = ma.masked
        self.v[1:-1] = (self.X[2:, :] - self.X[0:-2])/(2.0/self.frameRate)
        self.s = np.sqrt(np.sum(np.power(self.v, 2), axis=1))
        self.phi = np.arctan2(self.v[:, 1], self.v[:, 0])
        self.excludeBadFrames()

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
            bins = np.sqrt(np.sum(np.logical_not(self.badFrames)))

        plt.hist(self.s.compressed(), bins, normed=True)
        plt.xlabel('Speed (um/s)')
        plt.ylabel('Probability')
        plt.show()

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
