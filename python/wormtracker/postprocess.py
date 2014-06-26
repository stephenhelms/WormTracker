import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
#from numba import jit
from wormtracker import Logger

class WormTrajectoryPostProcessor:
    # bad frame settings
    filterByWidth = True
    filterByLength = True
    widthThreshold = (0.5, 1.5)
    lengthThreshold = (0.8, 1.2)

    # segment settings
    max_n_missing = 10
    max_d_um = 10
    max_segment_frames = 500
    min_segment_size = 150

    # head assignment settings
    headMinSpeed = 40  # Min <s> for head assignment by leading end
    headMinLeading = 1.3  # Min relative time leading for head
    headMinRelSpeed = 1.1  # Min relative end speed for head
    headMinRelBrightness = 0.2  # Min relative brightness for head

    def __init__(self, h5obj, strain, name):
        self.h5obj = h5obj
        self.strain = strain
        self.name = name
        self.h5ref = h5obj['worms'][strain][name]
        self.lengths = self.h5ref['length'][...]
        self.widths = self.h5ref['width'][...]
        self.frameRate = h5obj['/video/frameRate'][0]
        self.pixelsPerMicron = h5obj['/video/pixelsPerMicron'][0]
        self.maxFrameNumber = self.h5ref['time'].shape[0]
        self.nAngles = self.h5ref['posture'].shape[1]
        self.badFrames = np.zeros((self.maxFrameNumber,), dtype='bool')
        self.haveSkeleton = np.zeros((self.maxFrameNumber,), dtype='bool')
        self.skeleton = None
        self.posture = None
        self.length = None
        self.width = None
        self.t = self.h5ref['time'][...]
        self.X = ma.array(self.h5ref['centroid'][...] / self.pixelsPerMicron)
        self.Xhead = ma.zeros(self.X.shape)
        self.Xtail = ma.zeros(self.X.shape)
        self.v = ma.zeros(self.X.shape)
        self.s = ma.zeros((self.maxFrameNumber,))
        self.phi = ma.zeros((self.maxFrameNumber,))
        self.psi = ma.zeros((self.maxFrameNumber,))
        self.dpsi = ma.zeros((self.maxFrameNumber,))
        self.Ctheta = None
        self.ltheta = None
        self.vtheta = None

    def postProcess(self):
        Logger.logPrint( 'Identifying bad frames...')
        self.identifyBadFrames()
        Logger.logPrint( 'Extracting postural data...')
        self.extractPosturalData()
        Logger.logPrint( 'Fixing order of postural data...')
        self.fixPosturalOrdering()
        Logger.logPrint( 'Segmenting trajectory...')
        self.segment()
        Logger.logPrint( 'Assigning head...')
        self.assignHeadTail()
        Logger.logPrint( 'Ordering postural data head to tail...')
        self.orderHeadTail()
        Logger.logPrint( 'Calculating centroid motion variables...')
        self.calculateCentroidMeasurements()
        Logger.logPrint( 'Calculate postural measurements...')
        self.calculatePosturalMeasurements()

    def identifyBadFrames(self):
        badFrames = np.logical_or(self.lengths == 0,
                                  self.widths == 0)
        self.length = np.median(self.lengths[np.logical_not(badFrames)])
        self.width = np.median(self.widths[np.logical_not(badFrames)])
        if self.filterByWidth:
            badFrames = np.logical_or(badFrames,
                np.logical_or(self.widths < self.widthThreshold[0]*self.width,
                              self.widths > self.widthThreshold[1]*self.width))
        if self.filterByLength:
            badFrames = np.logical_or(badFrames,
                np.logical_or(self.lengths <
                              self.lengthThreshold[0]*self.length,
                              self.lengths >
                              self.lengthThreshold[1]*self.length))
        self.badFrames = badFrames

    def extractPosturalData(self):
        # import skeleton splines
        self.skeleton = self.h5ref['skeletonSpline'][...]
        self.posture = ma.array(self.h5ref['posture'][...])
        self.haveSkeleton = [np.any(skeleton > 0)
                             for skeleton in self.skeleton]

    # @jit
    @staticmethod 
    def skeletonDist(skeleton1, skeleton2):
        distEachPoint = np.sqrt(np.sum(np.power(skeleton1 -
                                                skeleton2, 2),
                                       axis=1))
        # return average distance per spline point
        return np.sum(distEachPoint)/skeleton1.shape[0]

    def fixPosturalOrdering(self):
        # compare possible skeleton orientations
        interframe_d = np.empty((self.maxFrameNumber, 2)) * np.NaN
        flipped = np.zeros((self.maxFrameNumber,), dtype=bool)
        nFromLastGood = np.empty((self.maxFrameNumber,)) * np.NaN

        for i in xrange(1, self.maxFrameNumber):
            # check whether there is a previous skeleton to compare
            if not self.haveSkeleton[i] or not np.any(self.haveSkeleton[:i]):
                continue

            ip = np.where(self.haveSkeleton[:i])[0][-1]  # last skeleton
            nFromLastGood[i] = i - ip
            interframe_d[i, 0] = self.skeletonDist(
                np.squeeze(self.skeleton[i, :, :]),
                np.squeeze(self.skeleton[ip, :, :]))
            # flipped orientation
            interframe_d[i, 1] = self.skeletonDist(
                np.flipud(np.squeeze(self.skeleton[i, :, :])),
                np.squeeze(self.skeleton[ip]))
            if interframe_d[i, 1] < interframe_d[i, 0]:
                # if the flipped orientation is better, flip the data
                flipped[i] = not flipped[ip]
            else:
                flipped[i] = flipped[ip]
        self.interframe_d = interframe_d

        # flip data appropriately
        sel = self.haveSkeleton and flipped
        self.skeleton[sel, :, :] = np.fliplr(np.squeeze(
            self.skeleton[sel, :, :]))
        self.posture[sel, :] = np.flipud(np.squeeze(self.posture[sel, :]))

    def segment(self):
        # break video into segments with matched skeletons
        max_d = self.max_d_um/self.pixelsPerMicron
        ii = 0
        segments = []
        while ii < self.maxFrameNumber:
            begin = ii
            ii += 1
            # Continue segment until >max_n_missing consecutive bad frames
            # are found, or >max_segment_frames are collected
            n_missing = 0
            last_missing = False
            while (ii < self.maxFrameNumber and
                   ii - begin < self.max_segment_frames and
                   (self.interframe_d[ii, 0] == np.NaN or
                    np.min(self.interframe_d[ii, :]) < max_d)):
                if not self.haveSkeleton[ii]:
                    n_missing += 1
                    last_missing = True
                    if n_missing > self.max_n_missing:
                        ii += 1
                        break
                else:
                    n_missing = 0
                    last_missing = False
                ii += 1
            segments.append([begin, ii])

        self.segments = [segment for segment in segments
                         if np.sum(self.haveSkeleton[segment[0]:segment[1]]) >
                         self.min_segment_size]

    def assignHeadTail(self):
        flipSegment = np.zeros((len(self.segments),), dtype='bool')
        segmentAssignMethod = -np.ones((len(self.segments),), dtype='int8')
        X = ma.array(self.h5ref['centroid'][...] / self.pixelsPerMicron)
        X[self.badFrames, :] = ma.masked
        dt = 1/self.frameRate
        (v, s, phi) = _getMotionVariables(X, dt)
        Xhead = np.squeeze(self.skeleton[:, 0, :] - self.h5ref['midpoint'][...])
        Xhead[self.badFrames, :] = ma.masked
        Xtail = np.squeeze(self.skeleton[:, -1, :] - self.h5ref['midpoint'][...])
        Xtail[self.badFrames, :] = ma.masked
        for i, segment in enumerate(self.segments):
            # method 1: head leads during movement
            # first check worm is actually moving significantly
            b = segment[0]
            e = segment[1]
            if (np.median(s[b:e].compressed()) >
                    self.headMinSpeed):
                # then calculate relative time moving in each direction
                phi_ends = np.zeros((e-b, 2))
                phi_ends[:, 0] = np.arctan2((Xhead[b:e, 1] - X[b:e, 1]),
                                            (Xhead[b:e, 0] - X[b:e, 0]))
                phi_ends[:, 1] = np.arctan2((Xtail[b:e, 1] - X[b:e, 1]),
                                            (Xtail[b:e, 0] - X[b:e, 0]))
                rphi_ends = np.cos(phi_ends.T - phi[b:e]).T
                first_leading = np.sum((rphi_ends[:, 0] >
                                        rphi_ends[:, 1]).compressed())
                last_leading = np.sum((rphi_ends[:, 1] >
                                       rphi_ends[:, 0]).compressed())
                if (max(first_leading, last_leading) /
                    min(first_leading, last_leading) >
                        self.headMinLeading):
                    segmentAssignMethod[i] = 1
                    if last_leading > first_leading:
                        flipSegment[i] = True
                    continue
            # method 2: head moves more than tail
            (vh, sh, phih) = _getMotionVariables(Xhead[b:e, :], dt)
            (vt, st, phit) = _getMotionVariables(Xtail[b:e, :], dt)
            mu_sh = np.mean(sh.compressed())
            mu_st = np.mean(st.compressed())
            if (max(mu_sh, mu_st) / min(mu_sh, mu_st) >
                    self.headMinRelSpeed):
                segmentAssignMethod[i] = 2
                if mu_st > mu_sh:
                    flipSegment[i] = True
                continue
            # method 3: head is brighter
            # this method isn't very reliable and isn't being used
        self.flipSegment = flipSegment
        self.segmentAssignMethod = segmentAssignMethod

    def orderHeadTail(self):
        orientationFixed = np.zeros((self.maxFrameNumber,), dtype='bool')
        for i, segment in enumerate(self.segments):
            if self.segmentAssignMethod[i] > 0:
                b = segment[0]
                e = segment[1]
                orientationFixed[b:e] = True
                if self.flipSegment[i]:
                    self.skeleton[b:e, :, :] = \
                        np.fliplr(self.skeleton[b:e, :, :])
                    self.posture[b:e, :] = \
                        np.fliplr(self.posture[b:e, :])
        self.orientationFixed = orientationFixed

    def calculateCentroidMeasurements(self):
        self.X[self.badFrames, :] = ma.masked
        self.v[1:-1] = (self.X[2:, :] - self.X[0:-2])/(2.0/self.frameRate)
        self.s = np.sqrt(np.sum(np.power(self.v, 2), axis=1))
        self.phi = np.arctan2(self.v[:, 1], self.v[:, 0])
        self.t[self.badFrames] = ma.masked
        self.X[self.badFrames, :] = ma.masked
        self.v[self.badFrames, :] = ma.masked
        self.s[self.badFrames] = ma.masked
        self.phi[self.badFrames] = ma.masked

    def calculatePosturalMeasurements(self):
        self.Xhead = ma.array(np.squeeze(self.skeleton[:, 0, :]))
        self.Xhead = ((self.Xhead + self.h5ref['boundingBox'][:, :2]) /
                      self.pixelsPerMicron)
        self.Xhead[np.logical_not(self.orientationFixed), :] = ma.masked
        self.Xtail = ma.array(np.squeeze(self.skeleton[:, -1, :]))
        self.Xtail = ((self.Xtail + self.h5ref['boundingBox'][:, :2]) /
                      self.pixelsPerMicron)
        self.Xtail[np.logical_not(self.orientationFixed), :] = ma.masked
        self.psi = np.arctan2(self.Xhead[:, 1]-self.X[:, 1],
                              self.Xhead[:, 0]-self.X[:, 0])
        self.dpsi = self.phi - self.psi
        self.psi[np.logical_not(self.orientationFixed)] = ma.masked
        self.dpsi[np.logical_or(np.logical_not(self.orientationFixed),
                                self.badFrames)] = ma.masked
        self.Xhead[np.logical_or(np.logical_not(self.orientationFixed),
                                 self.badFrames), :] = ma.masked
        self.Xtail[np.logical_or(np.logical_not(self.orientationFixed),
                                 self.badFrames), :] = ma.masked

        self.skeleton[np.logical_not(self.orientationFixed), :, :] = ma.masked
        self.posture[np.logical_not(self.orientationFixed), :] = ma.masked
        missing = np.any(self.posture.mask, axis=1)
        if np.all(missing):
            self.Ctheta = None
            self.ltheta = None
            self.vtheta = None
        else:
            posture = self.posture[~missing, :].T
            self.Ctheta = np.cov(posture)
            self.ltheta, self.vtheta = LA.eig(self.Ctheta)

    def store(self):
        # head assignment
        if 'segments' not in self.h5ref:
                self.h5ref.create_dataset('segments',
                                          (len(self.segments), 2),
                                          maxshape=(self.maxFrameNumber, 2),
                                          chunks=True,
                                          dtype='int')
        if len(self.segments) > self.h5ref['segments'].shape[0]:
            self.h5ref['segments'].resize((len(self.segments, 2)))
        self.h5ref['segments'][:len(self.segments), :] = \
            np.array(self.segments)
        self.h5ref['segments'][len(self.segments):, :] = -1

        if 'segmentAssignMethod' not in self.h5ref:
                self.h5ref.create_dataset('segmentAssignMethod',
                                          (len(self.segments),),
                                          maxshape=(self.maxFrameNumber,),
                                          chunks=True,
                                          dtype='int')
        if len(self.segments) > self.h5ref['segmentAssignMethod'].shape[0]:
            self.h5ref['segmentAssignMethod'].resize((len(self.segments,)))
        self.h5ref['segmentAssignMethod'][:len(self.segments)] = \
            self.segmentAssignMethod
        self.h5ref['segmentAssignMethod'][len(self.segments):] = 0

        if 'orientationFixed' not in self.h5ref:
                self.h5ref.create_dataset('orientationFixed',
                                          (self.maxFrameNumber,),
                                          dtype='bool')
        self.h5ref['orientationFixed'][...] = self.orientationFixed

        if 'badFrames' not in self.h5ref:
                self.h5ref.create_dataset('badFrames',
                                          (self.maxFrameNumber,),
                                          dtype='bool')
        self.h5ref['badFrames'][...] = self.badFrames
        self.h5ref['skeletonSpline'][...] = self.skeleton
        self.h5ref['posture'][...] = self.posture

        # centroid measurements
        if 'X' not in self.h5ref:
            self.h5ref.create_dataset('X',
                                      (self.maxFrameNumber, 2),
                                      dtype='f8')
        self.h5ref['X'][...] = self.X.filled(np.NaN)
        if 'v' not in self.h5ref:
            self.h5ref.create_dataset('v',
                                      (self.maxFrameNumber, 2),
                                      dtype='f8')
        self.h5ref['v'][...] = self.v.filled(np.NaN)
        if 's' not in self.h5ref:
            self.h5ref.create_dataset('s',
                                      (self.maxFrameNumber,),
                                      dtype='f8')
        self.h5ref['s'][...] = self.s.filled(np.NaN)
        if 'phi' not in self.h5ref:
            self.h5ref.create_dataset('phi',
                                      (self.maxFrameNumber,),
                                      dtype='f8')
        self.h5ref['phi'][...] = self.phi.filled(np.NaN)
        if 'psi' not in self.h5ref:
            self.h5ref.create_dataset('psi',
                                      (self.maxFrameNumber,),
                                      dtype='f8')
        self.h5ref['psi'][...] = self.psi.filled(np.NaN)
        if 'dpsi' not in self.h5ref:
            self.h5ref.create_dataset('dpsi',
                                      (self.maxFrameNumber,),
                                      dtype='f8')
        self.h5ref['dpsi'][...] = self.dpsi.filled(np.NaN)
        if 'Xhead' not in self.h5ref:
            self.h5ref.create_dataset('Xhead',
                                      (self.maxFrameNumber, 2),
                                      dtype='f8')
        self.h5ref['Xhead'][...] = self.Xhead.filled(np.NaN)
        if 'Xtail' not in self.h5ref:
            self.h5ref.create_dataset('Xtail',
                                      (self.maxFrameNumber, 2),
                                      dtype='f8')
        self.h5ref['Xtail'][...] = self.Xtail.filled(np.NaN)
        # postural measurements
        if 'avgLength' not in self.h5ref:
            self.h5ref.create_dataset('avgLength',
                                      (1,),
                                      dtype='f8')
        self.h5ref['avgLength'][...] = self.length
        if 'avgWidth' not in self.h5ref:
            self.h5ref.create_dataset('avgWidth',
                                      (1,),
                                      dtype='f8')
        self.h5ref['avgWidth'][...] = self.width
        if 'Ctheta' not in self.h5ref:
            self.h5ref.create_dataset('Ctheta',
                                      (self.nAngles, self.nAngles),
                                      maxshape=(100, 100),
                                      dtype='f8')
        if self.Ctheta is not None:
            self.h5ref['Ctheta'][...] = self.Ctheta
        if 'ltheta' not in self.h5ref:
            self.h5ref.create_dataset('ltheta',
                                      (self.nAngles,),
                                      maxshape=(100,),
                                      dtype='f8')
        if self.ltheta is not None:
            self.h5ref['ltheta'][...] = self.ltheta
        if 'vtheta' not in self.h5ref:
            self.h5ref.create_dataset('vtheta',
                                      (self.nAngles, self.nAngles),
                                      maxshape=(100, 100),
                                      dtype='f8')
        if self.vtheta is not None:
            self.h5ref['vtheta'][...] = self.vtheta


def _getMotionVariables(X, dt):
    v = ma.zeros(X.shape)
    v[0] = ma.masked
    v[-1] = ma.masked
    v[1:-1] = (X[2:, :] - X[0:-2])/(2.0*dt)
    s = np.sqrt(np.sum(np.power(v, 2), axis=1))
    phi = np.arctan2(v[:, 1], v[:, 0])
    return (v, s, phi)
