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
        self.phi = self.trajectory.getMaskedCentroid(self.trajectory.phi)
        self.psi = self.trajectory.getMaskedPosture(self.trajectory.psi)
        self.skeleton = ma.array(self.trajectory.skeleton[:, 1:-1, :])
        self.skeleton[~self.trajectory.orientationFixed, :, :] = ma.masked
        self.posture = self.trajectory.getMaskedPosture(self.trajectory.posture)
        self.pixelsPerMicron = self.trajectory.pixelsPerMicron
        self.frameRate = self.trajectory.frameRate
        self.quiverLength = 300.*self.pixelsPerMicron

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
        self.phiQuiver = self.axWorm.quiver(0,0,0,0, color='r',
                                            angles='xy', scale_units='xy',
                                            scale=1)
        self.psiQuiver = self.axWorm.quiver(0,0,0,0, color='b',
                                            angles='xy', scale_units='xy',
                                            scale=1)
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
        xc = self.X[frameNumber, 1]*self.pixelsPerMicron - bb[1]
        self.centroid.set_xdata(xc)
        yc = self.X[frameNumber, 0]*self.pixelsPerMicron - bb[0]
        self.centroid.set_ydata(yc)
        self.head.set_xdata(self.Xhead[frameNumber, 1]*self.pixelsPerMicron - bb[1])
        self.head.set_ydata(self.Xhead[frameNumber, 0]*self.pixelsPerMicron - bb[0])
        if not self.phi.mask[frameNumber]:
            self.phiQuiver.set_offsets([xc, yc])
            self.phiQuiver.set_UVC(
                np.cos(self.phi[frameNumber])*self.quiverLength,
                np.sin(self.phi[frameNumber])*self.quiverLength)
        else:
            self.phiQuiver.set_offsets([0,0])
            self.phiQuiver.set_UVC(0,0)
        if not self.psi.mask[frameNumber]:
            self.psiQuiver.set_offsets([xc, yc])
            self.psiQuiver.set_UVC(
                np.cos(self.psi[frameNumber])*self.quiverLength,
                np.sin(self.psi[frameNumber])*self.quiverLength)
        else:
            self.psiQuiver.set_offsets([0,0])
            self.psiQuiver.set_UVC(0,0)
        plt.title('%i: %.2f s'%(frameNumber,frameNumber/self.frameRate))

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
            interval = 1000./self.frameRate
        return animation.FuncAnimation(figure, self.plot,
                                       frames=frames,
                                       init_func=self.initialView,
                                       interval=interval)

    def showAnimation(self, frames=None, interval=None):
        self.getAnimation(frames=frames, interval=interval)
        plt.show()


class AnimatedWormPhaseSpaceWithImage:
    def __init__(self, trajectory, vtheta=None, v1=1, v2=2):
        self.trajectory = trajectory
        self.t = self.trajectory.t
        self.badFrames = self.trajectory.badFrames
        self.posture = self.trajectory.getMaskedPosture(self.trajectory.posture)
        self.X = self.trajectory.getMaskedCentroid(self.trajectory.X)
        self.skeleton = ma.array(self.trajectory.skeleton[:, 1:-1, :])
        self.skeleton[~self.trajectory.orientationFixed, :, :] = ma.masked
        self.frameRate = self.trajectory.frameRate
        self.pixelsPerMicron = self.trajectory.pixelsPerMicron
        if vtheta is None:
            self.vtheta = self.trajectory.vtheta
        else:
            self.vtheta = vtheta
        self.v1 = v1
        self.v2 = v2
        self.postureProj = ma.dot(self.posture, self.vtheta[:, [v1,v2]])

        self.nPast = 115*6
        self.showBadWormImages = False

    def initialView(self):
        self.axPhase = plt.subplot(1, 2, 2)
        self.phaseTraj, = self.axPhase.plot([], [], '-', color=(0.5,0.5,0.5))
        self.phasePoint, = self.axPhase.plot([], [], 'ro')
        plt.xlim((-10, 10))
        plt.xlabel('$a_'+str(self.v1)+'$')
        plt.ylim((-10, 10))
        plt.ylabel('$a_'+str(self.v2)+'$')
        self.axPhase.set_aspect(1)

        self.axWorm = plt.subplot(1, 2, 1)
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
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

    def plot(self, frameNumber):
        # phase plot
        first = frameNumber - self.nPast
        if first < 0:
            first = 0
        self.phaseTraj.set_xdata(self.postureProj[first:frameNumber, 0])
        self.phaseTraj.set_ydata(self.postureProj[first:frameNumber, 1])
        self.phasePoint.set_xdata(self.postureProj[frameNumber, 0])
        self.phasePoint.set_ydata(self.postureProj[frameNumber, 1])
        # worm plot
        im = self.getWormImage(frameNumber)
        self.imWorm.set_array(im)
        self.imWorm.set_extent((0, im.shape[1], 0, im.shape[0]))
        bb = self.trajectory.h5ref['boundingBox'][frameNumber,
                                                  :2]
        skel = self.trajectory.h5ref['skeleton'][frameNumber, :, :]
        empty = np.all(skel == 0, axis=1)
        if ~self.showBadWormImages and np.any(self.posture.mask[frameNumber, :]):
            self.skelLine.set_xdata(np.array([]))
            self.skelLine.set_ydata(np.array([]))
            self.postureSkel.set_offsets(np.array([[]]))
            self.postureSkel.set_array(np.array([]))
            self.centroid.set_xdata([])
            self.centroid.set_ydata([])
        else:
            self.skelLine.set_xdata(skel[~empty, 1])
            self.skelLine.set_ydata(skel[~empty, 0])
            self.postureSkel.set_offsets(np.fliplr(self.skeleton[frameNumber, :, :]))
            self.postureSkel.set_array(self.posture[frameNumber, :])
            self.postureSkel.set_clim(-1, 1)
            self.postureSkel.set_cmap(plt.get_cmap('PuOr'))
            xc = self.X[frameNumber, 1]*self.pixelsPerMicron - bb[1]
            self.centroid.set_xdata(xc)
            yc = self.X[frameNumber, 0]*self.pixelsPerMicron - bb[0]
            self.centroid.set_ydata(yc)
        plt.title('%i: %.2f s'%(frameNumber,frameNumber/self.frameRate))

    def getWormImage(self, frameNumber):
        bb = self.trajectory.h5ref['boundingBox'][frameNumber,
                                                  2:]

        if all(bb == 0):
            return np.zeros((100,100))
        if np.any(self.posture.mask[frameNumber, :]):
            if ~self.showBadWormImages:
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
            interval = 1000./self.frameRate
        return animation.FuncAnimation(figure, self.plot, blit=True,
                                       frames=frames,
                                       init_func=self.initialView)#,
                                       #interval=interval)

    def showAnimation(self, frames=None, interval=None):
        self.getAnimation(frames=frames, interval=interval)
        plt.show()