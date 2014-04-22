import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import cv2
from skimage import morphology, graph
import h5py
from subprocess import check_output
import sys
import wormimageprocessor

# Interactive script - replace VideoReader with the Python/OpenCV approach
# (only need a single frame anyway)
# Automated analysis, parallelized per worm (16/video in the new ones):

# NOTE: Matlab crop indices are flipped (y,x,h,w) relative to Numpy arrays


libavPath = 'C:\\libav\\bin\\'


class WormVideoRegion:
    """ Processes a region of a worm behavior experiment containing
        a single worm. """
    frameRate = 11.5
    frameSize = (2736, 2192)
    foodCircle = None

    croppedFilteredVideoFile = None
    thresholdedVideoFile = None

    def __init__(self, videoFile, imageProcessor, resultsStoreFile,
                 cropRegion, pixelSize,
                 resultsStorePath='/worms', outputPrefix='temp',
                 strainName='Unknown', wormName=''):
        self.videoFile = videoFile
        self.imageProcessor = imageProcessor
        self.resultsStoreFile = resultsStoreFile
        self.resultsStorePath = resultsStorePath
        self.outputPrefix = outputPrefix
        self.cropRegion = cropRegion
        self.pixelSize = pixelSize
        self.strainName = strainName
        self.wormName = wormName

    def process(self):
        """ Processes the video region. """
        self.generateCroppedFilteredVideo()
        self.generateThresholdedVideo()
        self.identifyWorm()

    def generateCroppedFilteredVideo(self):
        """ Crops and filters the video frames """
        if self.croppedFilteredVideoFile == '':
            self.croppedFilteredVideoFile = self.outputPrefix + '_cropped.avi'

        check_output([libavPath + 'avconv', '-i', self.videoFile, '-vf',
                      'crop=' + self._cropRegionForAvconv(), '-c:v',
                      'rawvideo', '-pix_fmt', 'yuv420p',
                      '-y', 'temp_' + self.croppedFilteredVideoFile])

        croppedVideo = cv2.VideoCapture()
        if croppedVideo.open('temp_' + self.croppedFilteredVideoFile):
            filteredVideoOut = cv2.VideoWriter()
            if filteredVideoOut.open(self.croppedFilteredVideoFile,
                                     cv2.cv.CV_FOURCC('Y', '8', '0',
                                                      '0'),
                                     self.frameRate,
                                     (self.cropRegion[3],
                                      self.cropRegion[4]),
                                     isColor=False):
                # loop through video frames
                success, frame = croppedVideo.read()
                while success:
                    framev = cv2.split(frame)  # split the channels
                    # filter frame: inverted black hat filter
                    filtered = \
                        self.imageProcessor.applyBackgroundFilter(framev[0])
                    # write frame to output
                    filteredVideoOut.write(filtered)
                    # read next video frame
                    success, frame = croppedVideo.read()
            else:
                raise Exception('Error opening filtered video for ' +
                                'writing in OpenCV.')
        else:
            raise Exception('Error opening filtered video in OpenCV.')
        # TODO: Delete temporary cropped video

    def _cropRegionForAvconv(self):
        return 'x=' + str(self.cropRegion[0]) + ':' + \
            'y=' + str(self.cropRegion[1]) + ':' + \
            'out_w=' + str(self.cropRegion[2]) + ':' + \
            'out_h=' + str(self.cropRegion[3])

    def generateThresholdedVideo(self):
        """ Thresholds all the filtered frames and applies
        morphological cleaning steps
        """
        if self.thresholdedVideoFile == '':
            self.thresholdedVideoFile = self.outputPrefix + '_thresholded.avi'

        filteredVideo = cv2.VideoCapture()
        if filteredVideo.open(self.croppedFilteredVideoFile):
            thresholdedVideoOut = cv2.VideoWriter()
            if thresholdedVideoOut.open(self.thresholdedVideoFile,
                                        cv2.cv.CV_FOURCC('Y', '8',
                                                         '0', '0'),
                                        self.frameRate,
                                        (self.cropRegion[3],
                                         self.cropRegion[4]),
                                        isColor=False):
                # loop through video frames
                success, frame = filteredVideo.read()
                while success:
                    framev = cv2.split(frame)  # split the channels
                    ip = self.imageProcessor
                    thresholded = ip.applyThreshold(framev[0])
                    cleaned = ip.applyMorphologicalCleaning(thresholded)
                    # write frame to output
                    thresholdedVideoOut.write(cleaned)
                    # read next video frame
                    success, frame = filteredVideo.read()
            else:
                raise Exception('Error opening filtered video for ' +
                                'writing in OpenCV.')
        else:
            raise Exception('Error opening filtered video in OpenCV.')
        # TODO: call('avconv','-i',self.croppedFilteredVideoFile,'-vf','?')
        # to figure out how to do this

    def identifyWorm(self):
        """ Loops through thresholded frames, identifies the likely worm,
            measures its properties, and stores the result in the data store
        """
        try:
            bwVideo = cv2.VideoCapture()
            if bwVideo.open(self.thresholdedVideoFile):
                grayVideo = cv2.VideoCapture()
                if grayVideo.open(self.croppedFilteredVideoFile):
                    # loop through video frames
                    count = 0
                    # read filtered video frame
                    bwSuccess, grayFrame = grayVideo.read()
                    # read thresholded video frame
                    graySuccess, bwFrame = bwVideo.read()
                    while bwSuccess and graySuccess:
                        # split the channels
                        bwFramev = cv2.split(bwFrame)
                        grayFramev = cv2.split(grayFrame)

                        ip = self.imageProcessor
                        # identify possible worms in image
                        # returns contours, areas
                        possibleWorms = ip.identifyPossibleWorms(bwFramev[0])
                        # likely worm is the largest area
                        likelyWorm = max(possibleWorms,
                                         key=lambda worm: worm[1])

                        # Create worm object which will measure
                        # the properties of the worm
                        worm = self.measureWorm(grayFramev[0],
                                                bwFramev[0],
                                                likelyWorm[0])

                        # write results to HDF5 store
                        worm.store(self.resultsStoreFile, count)

                        count += 1  # increment frame counter

                        # read next video frame
                        bwSuccess, grayFrame = grayVideo.read()
                        graySuccess, bwFrame = bwVideo.read()
                else:
                    raise Exception('Error opening filtered video ' +
                                    'in OpenCV.')
            else:
                raise Exception('Error opening thresholded video in ' +
                                'OpenCV.')
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise

    def measureWorm(self, grayFrame, bwFrame, wormContour):
        worm = WormImage(self, grayFrame, bwFrame, wormContour)
        worm.measureWorm()
        return worm

    def saveConfiguration(self):
        with h5py.File(self.resultsStoreFile) as f:
            pre = (self.resultsStorePath + '/' +
                   self.strainName + '/' +
                   self.wormName)
            # check whether datasets exist
            if pre not in f:
                g = f.create_group(pre)
                g.create_dataset('cropRegion', 4, dtype=int)
                g.create_dataset('foodCircle', 3, dtype=float)

            # write configuration
            g = f[pre]
            # strip directory info from file
            g['cropRegion'] = self.cropRegion
            g['foodCircle'] = self.foodCircle


class WormImage:
    boundingBox = None
    bwWormImage = None
    grayWormImage = None
    outlinedWormImage = None
    skeletonizedWormImage = None
    skeleton = None
    centroid = None
    midpoint = None
    width = None
    length = None
    posture = None

    def __init__(self, videoRegion, grayFrame, bwFrame, wormContour):
        self.videoRegion = videoRegion
        self.grayFrame = grayFrame
        self.bwFrame = bwFrame
        self.wormContour = wormContour

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
        cv2.drawContours(self.outlinedWormImage, self.wormContour, 0, 255,
                         thickness=1)
        self.outlinedWormImage = np.equal(self.outlinedWormImage, 255)

    def skeletonizeWorm(self):
        self.skeletonizedWormImage = morphology.skeletonize(self.bwWormImage)
        skeletonEnds = wormimageprocessor.find1Cpixels(
            self.skeletonizedWormImage)
        skeletonEndPts = cv2.findNonZero(np.uint8(skeletonEnds))
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
            self.skeleton = skeletonPts
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
            cx = int(moments['m10']/moments['m00'])  # cx = M10/M00
            cy = int(moments['m01']/moments['m00'])
            self.centroid = (cx, cy)
        else:
            self.centroid = []

    def calculateWidth(self):
        # approximate width as 2*shortest path to contour at midpoint
        mp = self.midpoint
        self.outlineWorm()
        cpts = np.float64(cv2.findNonZero(np.uint8(self.outlinedWormImage)))
        self.width = 2*min([np.sqrt((mp[0]-pt[0])**2 + (mp[1]-pt[1])**2)
                            for pt in cpts]) / self.videoRegion.pixelSize

    def calculatePosture(self):
        self.skeletonizeWorm()  # find skeleton and length
        pts = np.float64(self.skeleton)
        # distance along skeleton
        s = np.zeros((pts.shape[0], 1))
        for i in xrange(1, len(s)):
            s[i] = (np.sqrt((pts[i, 0]-pts[i-1, 0])**2 +
                            (pts[i, 1]-pts[i-1, 1])**2) +
                    s[i-1])
        # calculate length
        self.length = s[-1]/self.videoRegion.pixelSize
        # fit spline to skeleton
        fx = interpolate.InterpolatedUnivariateSpline(s/s[-1], pts[:, 0])
        fy = interpolate.InterpolatedUnivariateSpline(s/s[-1], pts[:, 1])
        # find midpoint
        self.midpoint = (fx(0.5), fy(0.5))
        # calculate body angles
        nAngles = self.videoRegion.imageProcessor.numberOfPosturePoints
        theta = np.zeros((nAngles, 1))
        sp = np.linspace(0, 1, nAngles+2)
        spi = np.array([fx(sp), fy(sp)]).transpose()
        for i in xrange(1, nAngles+1):
            theta[i-1] = np.arctan2((spi[i+1, 1]-spi[i-1, 1])/2.0,
                                    (spi[i+1, 0]-spi[i-1, 0])/2.0)
        theta = np.unwrap(theta)
        self.meanBodyAngle = np.mean(theta)
        self.posture = theta - self.meanBodyAngle

    def toRegionCoordinates(self, pts):
        if self.boundingBox is None:
            self.cropToWorm()  # crop the frame to the worm
        return [(pt[0] + self.boundingBox[0],
                 pt[1] + self.boundingBox[1])
                for pt in pts]

    def store(self, storeFile, index):
        raise NotImplemented()

    def plot(self):
        if self.bwWormImage is None:
            self.cropToWorm()
        if self.centroid is None:
            self.measureWorm()
        im = cv2.cvtColor(self.grayWormImage, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(im, self.wormContour, 0, (255, 0, 0))
        plt.imshow(im)
        plt.hold()
        plt.plot(self.skeletonPts[:, 0], self.skeletonPts[:, 1], 'r-')
        plt.scatter(self.skeletonPts[:, 0], self.skeletonPts[:, 1],
                    c=self.posture, cmap=plt.get_cmap('PuOr'))
        plt.plot(self.centroid[0], self.centroid[1], 'ro')
        plt.plot(self.midpoint[0], self.midpoint[1], 'rs')
