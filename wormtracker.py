import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage import morphology
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
            cx = int(moments['m10']/moments['m00'])  # cx = M10/M00
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
