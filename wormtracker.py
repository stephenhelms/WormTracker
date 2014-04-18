import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage import morphology
import h5py
from subprocess import check_output
import sys

# Interactive script - replace VideoReader with the Python/OpenCV approach
# (only need a single frame anyway)
# Automated analysis, parallelized per worm (16/video in the new ones):

# NOTE: Matlab crop indices are flipped (y,x,h,w) relative to Numpy arrays


class WormVideoRegion:
    """ Processes a region of a worm behavior experiment containing
        a single worm. """
    backgroundDiskRadius = 5
    backgroundThreshold = 0.9
    wormDiskSize = 2
    expectedWormLength = 1000
    expectedWormWidth = 50
    frameRate = 11.5
    numberOfPosturePoints = -1
    libavPath = 'C:\\libav\\bin\\'
    frameSize = (2736,2192)

    croppedFilteredVideoFile = ''
    thresholdedVideoFile = ''
    resultsStoreFile = ''

    def __init__(self, videoFile, resultsStoreFile, cropRegion, pixelSize,
                 outputPrefix='temp', strainName='Unknown', wormName=''):
        self.videoFile = videoFile
        self.resultsStoreFile = resultsStoreFile
        self.outputPrefix = outputPrefix
        self.cropRegion = cropRegion
        self.pixelSize = pixelSize
        self.strainName = strainName
        self.wormName = wormName

    def expectedWormLengthPixels(self):
        """ Returns the expected length of a worm in pixels """
        return self.expectedWormLength * self.pixelSize

    def expectedWormWidthPixels(self):
        """ Returns the expected width of a worm in pixels """
        return self.expectedWormWidth * self.pixelSize

    def expectedWormAreaPixels(self):
        """ Returns the expected area of a worm in pixels^2 """
        return (self.expectedWormLengthPixels() *
                self.expectedWormWidthPixels())

    def determineNumberOfPosturePoints(self):
        px = self.expectedWormLengthPixels()
        if px > 60:
            self.numberOfPosturePoints = 50
        elif px > 40:
            self.numberOfPosturePoints = 30
        elif px > 20:
            self.numberOfPosturePoints = 15
        else:  # not enough points to do postural analysis
            self.numberOfPosturePoints = 0

    def process(self):
        """ Processes the video region. """
        if self.numberOfPosturePoints < 0:
            self.determineNumberOfPosturePoints()
        self.generateCroppedFilteredVideo()
        self.generateThresholdedVideo()
        self.identifyWorm()

    def generateCroppedFilteredVideo(self):
        """ Uses libav to crop and apply a bottom hat filter to the video """
        if self.croppedFilteredVideoFile == '':
            self.croppedFilteredVideoFile = self.outputPrefix + '_cropped.avi'

        check_output([self.libavPath + 'avconv', '-i', self.videoFile, '-vf',
                      'crop=' + self.cropRegionForAvconv(), '-c:v',
                      'rawvideo', '-pix_fmt', 'yuv420p',
                      '-y', 'temp_' + self.croppedFilteredVideoFile])

        # Structuring element for bottom hat filtering the background
        bgSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (self.backgroundDiskRadius+1,
                                          self.backgroundDiskRadius+1))
        croppedVideo = cv2.VideoCapture()
        if croppedVideo.open('temp_' + self.croppedFilteredVideoFile):
            with cv2.VideoWriter() as filteredVideoOut:
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
                        filtered = (255 -
                                    cv2.morphologyEx(framev[0],
                                                     cv2.MORPH_BLACKHAT,
                                                     bgSE))
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

    def generateThresholdedVideo(self):
        """ Uses libav to threshold a video """
        if self.thresholdedVideoFile == '':
            self.thresholdedVideoFile = self.outputPrefix + '_thresholded.avi'

        with cv2.VideoCapture() as filteredVideo:
            if filteredVideo.open(self.croppedFilteredVideoFile):
                with cv2.VideoWriter() as thresholdedVideoOut:
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
                            # filter frame: inverted black hat filter
                            thresholded = cv2.threshold(framev[0],
                                                        255*self.threshold,
                                                        255, cv2.THRESH_BINARY)
                            # write frame to output
                            thresholdedVideoOut.write(thresholded)
                            # read next video frame
                            success, frame = filteredVideo.read()
                    else:
                        raise Exception('Error opening filtered video for ' +
                                        'writing in OpenCV.')
            else:
                raise Exception('Error opening filtered video in OpenCV.')
        # TODO: call('avconv','-i',self.croppedFilteredVideoFile,'-vf','?')
        # to figure out how to do this

    def cropRegionForAvconv(self):
        return 'x=' + str(self.cropRegion[0]) + ':' + \
            'y=' + str(self.cropRegion[1]) + ':' + \
            'out_w=' + str(self.cropRegion[2]) + ':' + \
            'out_h=' + str(self.cropRegion[3])

    def identifyWorm(self):
        try:
            with cv2.VideoCapture() as bwVideo:
                if bwVideo.open(self.thresholdedVideoFile):
                    with cv2.VideoCapture() as grayVideo:
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

                                # do connected component analysis
                                contours, hierarchy = cv2.findContours(
                                    bwFramev[0], cv2.RETR_CCOMP,
                                    cv2.CHAIN_APPROX_SIMPLE)

                                # find worm (largest connected component
                                # with plausible filled area)
                                areas = {i: cv2.contourArea(contour)
                                         for i, contour
                                         in enumerate(contours)}
                                n = self.expectedWormAreaPixels()
                                l = n*self.wormIdentificationThresholdRange[0]
                                u = n*self.wormIdentificationThresholdRange[1]
                                likelyWorm = max((idx for idx in areas
                                                 if areas[idx] > l and
                                                 areas[idx] < u),
                                                 key=lambda idx: areas[idx])

                                # Create worm object which will measure
                                # the properties of the worm
                                worm = self.measureWorm(grayFramev[0],
                                                        bwFramev[0],
                                                        likelyWorm)

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


class WormImage:
    boundingBox = []
    bwWormImage = []
    grayWormImage = []
    outlinedWormImage = []
    skeletonizedWormImage = []
    skeleton = []
    centroid = []
    midpoint = []
    width = []
    length = []
    posture = []

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
            self.boundingBox[0]:self.boundingBox[0]+self.boundingBox[2],
            self.boundingBox[1]:self.boundingBox[1]+self.boundingBox[3]]
        self.grayWormImage = self.grayFrame[
            self.boundingBox[0]:self.boundingBox[0]+self.boundingBox[2],
            self.boundingBox[1]:self.boundingBox[1]+self.boundingBox[3]]

    def outlineWorm(self):
        raise NotImplemented()

    def skeletonizeWorm(self):
        im = morphology.skeletonize(bwWormImage > 0)

    def measureWorm(self):
        # make sure the frame has been cropped
        if bwWormImage == [] or grayWormImage == []:
            cropToWorm()  # crop the frame to the worm

        calculateCentroid()  # measure centroid
        calculateWidth()  # measure width
        calculateLength()  # measure length
        calculatePosture()  # measure body angles
        # (store everything in HDF5)

    def calculateCentroid(self):
        moments = cv2.moments(self.wormContour)
        if moments['m00'] != 0:  # only calculate if there is a non-zero area
            cx = int(moments['m10']/moments['m00'])	 # cx = M10/M00
            cy = int(moments['m01']/moments['m00'])
            self.centroid = (cx, cy)
        else:
            self.centroid = []

    def calculateWidth(self):
        raise NotImplemented()

    def calculateLength(self):
        raise NotImplemented()

    def calculatePosture(self):
        raise NotImplemented()

    def store(self, storeFile, index):
        raise NotImplemented()
