import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from skimage import morphology, graph
import h5py
from subprocess import check_output
import sys
import os
import wormtracker.wormimageprocessor as wp
import roitools
import time
import multiprocessing

# NOTE: Matlab crop indices are flipped (y,x,h,w) relative to Numpy arrays

libavPath = 'C:\\libav\\bin\\'

class Logger: 
    @staticmethod
    def logPrint (line):
        #if hasattr(os, 'getppid'):  # only available on Unix
        #    parentPid=os.getppid()
        pid=os.getpid()
        print '['+str(pid)+']:'+line 
        
class WormVideo:
    def __init__(self, videoFile, storeFile='temp.h5',
                 videoInfoStorePath='/video',
                 resultsStorePath='/worms',
                 logFile='temp.log',
                 numberOfRegions=16, allSameStrain=True,
                 referenceDistance=25000):
        self.imageProcessor = wp.WormImageProcessor()
        self.firstFrame = None
        self.pixelsPerMicron = None
        self.regions = []
        self.frameSize = None
        self.nFrames = None
        self.videoFile = videoFile
        self.logFile=videoFile+'_.log'
        self.numberOfRegions = numberOfRegions
        self.allSameStrain = allSameStrain
        self.referenceDistance = referenceDistance
        self.storeFile = storeFile
        self.videoInfoStorePath = videoInfoStorePath
        self.resultsStorePath = resultsStorePath


    def readFirstFrame(self, askForFrameRate=True):
        video = cv2.VideoCapture()
        if video.open(self.videoFile):
            success, firstFrame = video.read()
            if not success:
                raise Exception("Couldn't read video")
            else:
                firstFrameChannels = cv2.split(firstFrame)
                self.firstFrame = firstFrameChannels[0]
                self.frameSize = self.firstFrame.shape
                self.nFrames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
                Logger.logPrint('Video has ' + str(self.nFrames) + ' frames.')
                if askForFrameRate:
                    frameRate = video.get(cv2.cv.CV_CAP_PROP_FPS)
                    Logger.logPrint('Video reports ' + str(frameRate) + ' fps.')
                    self.imageProcessor.frameRate = \
                        float(raw_input('Enter correct frame rate:'))

        else:
            raise Exception("Couldn't open video")

    def defineRegions(self):
        if self.firstFrame is None:
            self.readFirstFrame()

        # show first frame and ask user to select regions
        if self.allSameStrain:
            strain = raw_input("Enter the strain name:")
        regions = []

        def drawAllRegions():
            plt.imshow(self.firstFrame, cmap=plt.gray())
            ax = plt.gca()
            for region in regions:
                # draw region box
                rect = Rectangle((region[0][0], region[0][1]), region[0][2],
                                 region[0][3], color='k', fill=False)
                ax.add_patch(rect)
                # label region in lower left corner
                plt.text(region[0][0]+5, region[0][1]+region[0][3],
                         region[1] + ' ' + region[2])

        for i in xrange(self.numberOfRegions):
            raw_input("Select region " + str(i) + " on the figure...")
            drawAllRegions()  # show all regions already picked
            sel = roitools.RectangleRegionSelector()  # track selection
            plt.show()  # request user to select region on figure
            if not self.allSameStrain:
                strain = raw_input("Enter the strain name: ")
            wormName = raw_input("Enter the worm ID " +
                                 "(press enter to use the region number): ")
            if wormName is "":
                wormName = str(i+1)
            regions.append((sel.asXYWH(), strain, wormName))
        # show all picked regions
        drawAllRegions()
        plt.show()

        self.regions = []  # remove old regions, if any
        # add regions
        for region in regions:
            self.addRegion(tuple([int(round(x)) for x in region[0]]),
                           region[1], region[2])

    def updateVideoFile(self, videoFile):
        self.videoFile = videoFile
        self.getNumberOfFrames()
        for region in self.regions:
            region.videoFile = videoFile
            region.nFrames = self.nFrames

    def updateStoreFile(self, storeFile):
        self.storeFile = storeFile
        for region in self.regions:
            region.resultsStoreFile = storeFile

    def getNumberOfFrames(self):
        video = cv2.VideoCapture()
        if video.open(self.videoFile):
            success, firstFrame = video.read()
            if not success:
                raise Exception("Couldn't read video")
            else:
                self.nFrames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
                for region in self.regions:
                    region.nFrames = self.nFrames

    def addRegion(self, regionBounds, strain, name):
        """Adds the video region containing one worm.

        regionBounds: (x,y,w,h)
        strain: The strain name
        name: The worm identifier
        """
        wr = WormVideoRegion(self.videoFile, self.imageProcessor,
                             self.storeFile,
                             regionBounds, self.pixelsPerMicron,
                             resultsStorePath=self.resultsStorePath,
                             strainName=strain,
                             wormName=name)
        wr.nFrames = self.nFrames
        self.regions.append(wr)
        return wr

    def determinePixelSize(self):
        if self.firstFrame is None:
            self.readFirstFrame()
        raw_input("Draw a " + str(self.referenceDistance) +
                  " um line on the figure...")
        plt.imshow(self.firstFrame, cmap=plt.gray())
        sel = roitools.LineRegionSelector()
        plt.show()
        self.imageProcessor.pixelSize = (sel.distance() /
                                         self.referenceDistance)
        self.imageProcessor.determineNumberOfPosturePoints()
        WormVideo.logPrint("The pixel size is " + str(1.0/self.imageProcessor.pixelSize) + " um/px.")

    def defineFoodRegions(self):
        if self.firstFrame is None:
            self.readFirstFrame()
        raw_input("Draw a circle covering the food for each region...")
        for region in self.regions:
            crop = region.cropRegion
            plt.imshow(wp.cropImageToRegion(self.firstFrame, crop),
                       plt.gray())
            plt.title(region.strainName + " " + region.wormName)
            sel = roitools.CircleRegionSelector()
            plt.show()
            region.foodCircle = sel.asXYR()

    def testBackgroundFilter(self):
        if self.firstFrame is None:
            self.readFirstFrame()
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(self.firstFrame, cmap=plt.gray())
        plt.title('Original Frame')
        plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
        plt.imshow(self.imageProcessor.applyBackgroundFilter(self.firstFrame),
                   cmap=plt.gray())
        plt.title('Background Filtered')
        plt.show()

    def testThreshold(self):
        if self.firstFrame is None:
            self.readFirstFrame()
        ax1 = plt.subplot(1, 2, 1)
        filtered = self.imageProcessor.applyBackgroundFilter(self.firstFrame)
        plt.imshow(filtered, cmap=plt.gray())
        plt.title('Background Filtered Frame')
        plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
        plt.imshow(self.imageProcessor.applyThreshold(filtered),
                   cmap=plt.gray())
        plt.title('Thresholded')
        plt.show()

    def testMorphologicalCleaning(self, regionIdx):
        plt.figure()
        region = self.regions[regionIdx]
        ip = self.imageProcessor
        cropped = wp.cropImageToRegion(self.firstFrame, region.cropRegion)
        filtered = ip.applyBackgroundFilter(cropped)
        ax1 = plt.subplot(1,2,1)
        thresholded = ip.applyThreshold(filtered)
        plt.imshow(thresholded, plt.gray())
        plt.subplot(1,2,2, sharex=ax1, sharey=ax1)
        cleaned = ip.applyMorphologicalCleaning(thresholded)
        plt.imshow(cleaned, plt.gray())
        plt.show()

    def testWormIdentification(self):
        plt.figure()
        for i, region in enumerate(self.regions):
            plt.subplot(4, np.ceil(np.float64(self.numberOfRegions)/4.0), i+1)
            ip = self.imageProcessor
            cropped = wp.cropImageToRegion(self.firstFrame, region.cropRegion)
            filtered = ip.applyBackgroundFilter(cropped)
            thresholded = ip.applyThreshold(filtered)
            cleaned = ip.applyMorphologicalCleaning(thresholded)
            possibleWorms = ip.identifyPossibleWorms(cleaned)
            if len(possibleWorms) > 0:
                likelyWorm = max(possibleWorms, key=lambda worm: worm[1])
                if likelyWorm is not None:
                    try:
                        wormImage = WormImage(region, filtered, cleaned,
                                              likelyWorm[0])
                        wormImage.measureWorm()
                        wormImage.plot(bodyPtMarkerSize=30)
                    except(Exception) as e:
                        Logger.logPrint('Error in {0} {1}: {2}'.format(region.strainName,
                                                             region.wormName+':'+
                                                             str(e)))
            plt.title(region.strainName + ' ' + region.wormName)
        plt.show()

    def saveConfiguration(self):
        if os.path.isfile(self.storeFile):
            mode = 'r+'
        else:
            mode = 'w'
        with h5py.File(self.storeFile, mode) as f:
            pre = self.videoInfoStorePath
            # check whether datasets exist
            if pre not in f:
                f.create_group(pre)
            g = f[pre]
            if 'videoFile' not in g:
                dt = h5py.special_dtype(vlen=str)
                g.create_dataset('videoFile', (1,), dtype=dt)

            # write configuration
            # strip directory info from file
            path, fileName = os.path.split(self.videoFile)
            g['videoFile'][...] = fileName
            # save imaging configuration
            self.imageProcessor.saveConfiguration(self.storeFile,
                                                  self.videoInfoStorePath)
            # save information for each region
            for region in self.regions:
                region.saveConfiguration()

    def processRegions(self):
        self.saveConfiguration()
        Logger.logPrint('Processing regions of video...');
        for i, region in enumerate(self.regions):
            Logger.logPrint('Processing region ' + str(i) + ' of ' + str(len(self.regions)))
            tStart = time.clock()
            region.process()
            tStop = time.clock()
            tDuration = (tStop - tStart) / 60.0
            Logger.logPrint('Analysis of region took ' + str(tDuration) + ' min.')

    def processRegionsParallel(self):
        self.saveConfiguration()
        pool = multiprocessing.Pool()  # use as many CPU cores as you can
        #queue = multiprocessing.Queue()  # for messages

        # start a job for each region
        queue = multiprocessing.Queue()

        # split output into different files
        outputFiles = {}
        for region in self.regions:
            region.resultsStoreFile = (region.strainName + '_' + region.wormName +
                                       '_' + self.storeFile)
            outputFiles[region] = region.resultsStoreFile
        
        result = pool.map_async(_videoProcessRegionParallel, self.regions)
        #jobs = []
        #for region in self.regions:
        #    jobs.append(pool.apply_async(_videoProcessRegionParallel,
        #                                 region, queue))

        #while any(not job.ready() for job in jobs) and not queue.empty():
        #    if not queue.empty():
        #        logPrint queue.get()
        #    time.sleep(1)

        Logger.logPrint(result.get())
        pool.close()
        pool.join()
        Logger.logPrint('Finished analyzing all regions')
        # TODO: merge the output files

_queue = multiprocessing.Queue()

def _videoProcessRegionParallel(region):
    return region.processParallel(_queue)


class WormVideoRegion:
    """ Processes a region of a worm behavior experiment containing
        a single worm. """
    def __init__(self, videoFile, imageProcessor, resultsStoreFile,
                 cropRegion, pixelSize,
                 resultsStorePath='/worms', outputPrefix=None,
                 strainName='Unknown', wormName=''):
        self.frameRate = 11.5
        self.frameSize = (2736, 2192)
        self.foodCircle = None
        self.videoFile = videoFile
        self.imageProcessor = imageProcessor
        self.resultsStoreFile = resultsStoreFile
        self.resultsStorePath = resultsStorePath
        self.cropRegion = cropRegion
        self.pixelSize = pixelSize
        self.strainName = strainName
        self.wormName = wormName
        if outputPrefix is None:
            outputPrefix = self.strainName + '_' + self.wormName + '_'
        self.outputPrefix = outputPrefix

    def process(self):
        """ Processes the video region. """
        self.saveConfiguration()
        video = cv2.VideoCapture()
        ii = 0  # frame counter
        if video.open(self.videoFile):
            # loop through video frames
            success, frame = video.read()
            while success:
                framev = np.squeeze(frame[:, :, 0])  # split the channels
                tStart = time.clock()
                self.processFrame(framev, ii)
                tStop = time.clock()
                tDuration = (tStop - tStart)
                Logger.logPrint('Processing frame {0} of {1} took {2} s.'.format(
                    str(ii+1), str(self.nFrames), str(tDuration)))
                # read next video frame
                success, frame = video.read()
                ii = ii+1
        else:
            raise Exception('Error opening video in OpenCV.')

    def processFrame(self, frame, idx):
        # crop frame
        cFrame = wp.cropImageToRegion(frame, self.cropRegion)
        # filter frame
        fFrame = self.imageProcessor.applyBackgroundFilter(cFrame)
        # threshold frame
        tFrame = self.imageProcessor.applyThreshold(fFrame)
        # morphological cleaning
        clFrame = self.imageProcessor.applyMorphologicalCleaning(tFrame)
        # worm identification
        ip = self.imageProcessor
        # identify possible worms in image
        # returns contours, areas
        possibleWorms = ip.identifyPossibleWorms(clFrame)
        if (possibleWorms is not None and len(possibleWorms) > 0):
            # likely worm is the largest area
            likelyWorm = max(possibleWorms, key=lambda worm: worm[1])
        else:
            likelyWorm = None

        if likelyWorm is not None:
            # Create worm object which will measure
            # the properties of the worm
            try:
                worm = self.measureWorm(fFrame,
                                        clFrame,
                                        likelyWorm[0])

                # write results to HDF5 store
                pre = (self.resultsStorePath + '/' +
                       self.strainName + '/' +
                       self.wormName)
                worm.store(self.resultsStoreFile,
                           pre, idx)
            except(Exception) as e:
                Logger.logPrint('Error in {0} {1} frame {2} analyzing worm: {3}'.format(
                    self.strainName, self.wormName, str(idx), str(e)))

    def measureWorm(self, grayFrame, bwFrame, wormContour):
        worm = WormImage(self, grayFrame, bwFrame, wormContour)
        worm.measureWorm()
        return worm

    def saveConfiguration(self):
        if os.path.isfile(self.resultsStoreFile):
            mode = 'r+'
        else:
            mode = 'w'
        with h5py.File(self.resultsStoreFile, mode) as f:
            pre = (self.resultsStorePath + '/' +
                   self.strainName + '/' +
                   str(self.wormName))
            # check whether datasets exist
            if pre not in f:
                f.create_group(pre)
            g = f[pre]
            if 'cropRegion' not in g:
                g.create_dataset('cropRegion', (4,), dtype='int32')
                g.create_dataset('foodCircle', (3,), dtype='float64')

            # write configuration
            g['cropRegion'][...] = self.cropRegion
            g['foodCircle'][...] = self.foodCircle

            # create worm observation datasets
            n = self.nFrames
            if 'boundingBox' not in g:
                g.create_dataset('boundingBox', (n, 4), dtype='int32')
                g.create_dataset('bwWormImage', (n, 150, 150),
                                 maxshape=(n, None, None),
                                 chunks=(1,19,19),
                                 compression='gzip', dtype='b')
                g.create_dataset('grayWormImage', (n, 150, 150),
                                 maxshape=(n, None, None),
                                 chunks=(1,19,19),
                                 compression='gzip', dtype='uint8')
                g.create_dataset('skeleton', (n, 50, 2),
                                 maxshape=(n, 200, 2),
                                 chunks=(1,50,2), dtype='int32')
                g.create_dataset('skeletonSpline',
                                 (n, self.imageProcessor.numberOfPosturePoints, 2),
                                 maxshape=(n, 100, 2),
                                 chunks=(1,10,2), dtype='float64')
                g.create_dataset('centroid', (n, 2), dtype='float64')
                g.create_dataset('midpoint', (n, 2), dtype='float64')
                g.create_dataset('width', (n,), dtype='float64')
                g.create_dataset('length', (n,), dtype='float64')
                g.create_dataset('meanBodyAngle', (n,), dtype='float64')
                g.create_dataset('posture', (n, self.imageProcessor.numberOfPosturePoints),
                                 maxshape=(n, 100),
                                 chunks=(1, 10), dtype='float64')
                g.create_dataset('wormContour', (n, 2, 1, 2),
                                 maxshape=(n, None, 1, 2),
                                 chunks=(1, 10, 1, 2),
                                 fillvalue=-1,
                                 dtype='int32')
                g.create_dataset('time', (n,), dtype='float64')
                g.create_dataset('badSkeletonization', (n,), dtype='b')
                g.create_dataset('crossedWorm', (n,), dtype='b')

            rate = self.imageProcessor.frameRate
            g['time'][...] = np.float64(range(n))/rate


class WormImage:
    # default plot variables
    smoothing = 0.05
    outlineColor = (255, 255, 0)
    skeletonColor = 'y'
    postureColormap = plt.get_cmap('PuOr')
    centroidColor = 'r'
    midpointColor = 'r'

    def __init__(self, videoRegion, grayFrame, bwFrame, wormContour):
        self.videoRegion = videoRegion
        self.grayFrame = grayFrame
        self.bwFrame = bwFrame
        self.wormContour = wormContour

        # image data and measurements
        self.boundingBox = None
        self.bwWormImage = None
        self.grayWormImage = None
        self.outlinedWormImage = None
        self.skeletonizedWormImage = None
        self.skeleton = None
        self.skeletonSpline = None
        self.centroid = None
        self.midpoint = None
        self.width = None
        self.length = None
        self.posture = None
        self.meanBodyAngle = None
        self.badSkeletonization = False
        self.crossedWorm = False

    def cropToWorm(self):
        """ crop filtered and thresholded frames to worm """
        # measure bounding box
        self.boundingBox = cv2.boundingRect(self.wormContour)  # x,y,w,h

        # crop frame
        self.bwWormImage = self.bwFrame[
            self.boundingBox[1]:self.boundingBox[1]+self.boundingBox[3],
            self.boundingBox[0]:self.boundingBox[0]+self.boundingBox[2]]
        self.grayWormImage = self.grayFrame[
            self.boundingBox[1]:self.boundingBox[1]+self.boundingBox[3],
            self.boundingBox[0]:self.boundingBox[0]+self.boundingBox[2]]

    def outlineWorm(self):
        self.outlinedWormImage = np.zeros(self.bwWormImage.shape,
                                          dtype=np.uint8)
        cv2.drawContours(self.outlinedWormImage,
                         [self.toCroppedCoordinates(self.wormContour).astype('int')],
                         0, 255, thickness=1)
        self.outlinedWormImage = np.equal(self.outlinedWormImage, 255)

    def skeletonizeWorm(self):
        self.skeletonizedWormImage = morphology.skeletonize(self.bwWormImage)
        skeletonEnds = wp.find1Cpixels(self.skeletonizedWormImage)
        skeletonEndPts = cv2.findNonZero(np.uint8(skeletonEnds))
        if skeletonEndPts is None:
            skeletonEndPts = []
        nEndPts = len(skeletonEndPts)
        if nEndPts < 2:  # skeleton is a cirle (Omega turn)
            self.badSkeletonization = True
            self.crossedWorm = True
        elif nEndPts > 2:  # skeleton has spurs
            self.badSkeletonization = True
        else:
            skeletonInverted = np.logical_not(self.skeletonizedWormImage)
            skeletonPts, cost = \
                graph.route_through_array(np.uint8(skeletonInverted),
                                          np.flipud(skeletonEndPts[0][0]),
                                          np.flipud(skeletonEndPts[1][0]),
                                          geometric=True)
            self.skeleton = np.array([[pt[0], pt[1]] for pt in skeletonPts])
            self.badSkeletonization = False

    def measureWorm(self):
        # make sure the frame has been cropped
        if self.bwWormImage is None or self.grayWormImage is None:
            self.cropToWorm()  # crop the frame to the worm

        self.calculateCentroid()  # measure centroid
        self.calculatePosture()  # measure length, midpoint, and body angles
        self.calculateWidth()  # measure width

    def calculateCentroid(self):
        moments = cv2.moments(self.wormContour)
        if moments['m00'] != 0:  # only calculate if there is a non-zero area
            cx = float(moments['m10'])/float(moments['m00'])  # cx = M10/M00
            cy = float(moments['m01'])/float(moments['m00'])
            self.centroid = self.toRegionCoordinates( \
                np.flipud(self.toCroppedCoordinates([cx, cy])))
        else:
            self.centroid = None

    def calculateWidth(self):
        if self.badSkeletonization:
            return
        # approximate width as 2*shortest path to contour at midpoint
        mp = np.flipud(self.toCroppedCoordinates(self.midpoint))
        self.outlineWorm()
        cpts = np.float64(cv2.findNonZero(np.uint8(self.outlinedWormImage)))
        self.width = (min(np.sqrt(np.sum(np.float64(cpts - mp)**2, axis=2)))
                      * 2.0 / self.videoRegion.imageProcessor.pixelSize)

    def calculatePosture(self):
        self.skeletonizeWorm()  # find skeleton and length
        if self.badSkeletonization:
            self.skeleton = np.zeros((0, 2))
            self.skeletonSpline = np.zeros((0, 2))
            self.posture = np.zeros((0,))
            return
        pts = np.float64(self.skeleton)
        # distance along skeleton
        s = np.zeros((pts.shape[0], 1))
        for i in xrange(1, len(s)):
            s[i] = (np.sqrt((pts[i, 0]-pts[i-1, 0])**2 +
                            (pts[i, 1]-pts[i-1, 1])**2) +
                    s[i-1])
        # calculate length
        self.length = s[-1]/self.videoRegion.imageProcessor.pixelSize
        # fit spline to skeleton
        fx = interpolate.UnivariateSpline(s/s[-1], pts[:, 0],
                                          s=self.smoothing*pts.shape[0])
        fy = interpolate.UnivariateSpline(s/s[-1], pts[:, 1],
                                          s=self.smoothing*pts.shape[0])
        # find midpoint
        self.midpoint = self.toRegionCoordinates((fx(0.5), fy(0.5)))
        # calculate body angles
        nAngles = self.videoRegion.imageProcessor.numberOfPosturePoints
        theta = np.zeros(nAngles)
        sp = np.linspace(0, 1, nAngles+2)
        spi = np.array([fx(sp), fy(sp)]).transpose()
        self.skeletonSpline = spi
        for i in xrange(1, nAngles+1):
            theta[i-1] = np.arctan2((spi[i+1, 1]-spi[i-1, 1])/2.0,
                                    (spi[i+1, 0]-spi[i-1, 0])/2.0)
        theta = np.unwrap(theta)
        self.meanBodyAngle = np.mean(theta)
        self.posture = theta - self.meanBodyAngle

    def toCroppedCoordinates(self, pts):
        if self.boundingBox is None:
            self.cropToWorm()
        return pts - np.array(self.boundingBox[0:2], 'f8')

    def toRegionCoordinates(self, pts):
        if self.boundingBox is None:
            self.cropToWorm()  # crop the frame to the worm
        return pts + np.array(self.boundingBox[0:2], 'f8')

    def store(self, storeFile, storePath, index):
        if os.path.isfile(storeFile):
            mode = 'r+'
        else:
            mode = 'w'
        with h5py.File(storeFile, mode) as f:
            pre = storePath
            # check whether datasets exist
            f.require_group(pre)
            g = f[pre]

            # write configuration
            n = self.videoRegion.nFrames
            g['boundingBox'][index, :] = np.array(self.boundingBox)
            g['centroid'][index, :] = self.centroid
            g['badSkeletonization'][index] = self.badSkeletonization
            g['crossedWorm'][index] = self.crossedWorm
            if not self.badSkeletonization:
                g['midpoint'][index, :] = self.midpoint
                g['width'][index] = self.width
                g['length'][index] = self.length
                g['meanBodyAngle'][index] = self.meanBodyAngle

                s = self.skeleton.shape
                if g['skeleton'].shape[1:] != s:
                    g['skeleton'].resize((n,
                                          max(g['skeleton'].shape[1],
                                              s[0]),
                                          2))
                g['skeleton'][index, :s[0], :] = self.skeleton

                s = self.skeletonSpline.shape
                if g['skeletonSpline'].shape[1:] != s:
                    g['skeletonSpline'].resize((n,
                                                max(g['skeletonSpline'].shape[1],
                                                s[0]),
                                                2))
                g['skeletonSpline'][index, :s[0], :] = self.skeletonSpline

                s = self.posture.shape
                if g['posture'].shape[1:] != s:
                    g['posture'].resize((n,
                                         max(g['posture'].shape[1],
                                             s[0])))
                g['posture'][index, :s[0]] = self.posture

            s = self.bwWormImage.shape
            if g['bwWormImage'].shape[1:] != s:
                g['bwWormImage'].resize((n,
                                        max(g['bwWormImage'].shape[1],
                                            s[0]),
                                        max(g['bwWormImage'].shape[2],
                                            s[1])))
            g['bwWormImage'][index, :s[0], :s[1]] = self.bwWormImage

            s = self.grayWormImage.shape
            if g['grayWormImage'].shape[1:] != s:
                g['grayWormImage'].resize((n,
                                           max(g['grayWormImage'].shape[1],
                                               s[0]),
                                           max(g['grayWormImage'].shape[2],
                                               s[1])))
            g['grayWormImage'][index, :s[0], :s[1]] = self.grayWormImage
            
            s = self.wormContour.shape
            if g['wormContour'].shape[1:] != s:
                g['wormContour'].resize((n,
                                         max(g['wormContour'].shape[1],
                                             s[0]),
                                         1,
                                         2))
            g['wormContour'][index, :s[0], :, :] = self.wormContour

    def plot(self, bodyPtMarkerSize=100):
        if self.bwWormImage is None:
            self.cropToWorm()
        if self.centroid is None:
            self.measureWorm()
        im = cv2.cvtColor(cv2.normalize(self.grayWormImage,
                                        alpha=0,
                                        beta=255,
                                        norm_type=cv2.NORM_MINMAX),
                          cv2.COLOR_GRAY2RGB)
        cv2.drawContours(im,
                         [self.toCroppedCoordinates(self.wormContour)],
                         0, self.outlineColor)
        plt.imshow(im, interpolation='none')
        plt.hold(True)
        if self.centroid is not None:
            c = self.toCroppedCoordinates(self.centroid)
            plt.plot(c[1], c[0], 'o', ms=12,
                 color=self.centroidColor)
        if not self.badSkeletonization:
            plt.plot(self.skeleton[:, 1], self.skeleton[:, 0], '-',
                     color=self.skeletonColor)
            plt.scatter(self.skeletonSpline[1:-1, 1], self.skeletonSpline[1:-1, 0],
                        c=self.posture, cmap=self.postureColormap,
                        s=bodyPtMarkerSize)
            m = self.toCroppedCoordinates(self.midpoint)
            plt.plot(m[1], m[0], 's', ms=12,
                     color=self.midpointColor)


if __name__ == '__main__':
    multiprocessing.freeze_support()
