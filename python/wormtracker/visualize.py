import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import itertools
import collections
import cv2


class AnimatedWormTrajectoryWithImage:
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
        bb = self.trajectory.h5ref['boundingBox'][frameNumber,
                                                  2:]

        if all(bb == 0):
            return np.zeros((100,100))
        im = self.trajectory.h5ref['grayWormImage'][frameNumber,
                                                    :bb[1],
                                                    :bb[0]]
        im = cv2.normalize(im, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX)
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