import cv2
from skimage import segmentation
import numpy as np
from scipy import ndimage
from scipy.ndimage import filters as nf
import h5py

"""
Black hat filter on a 6MP frame takes:
0.015 s in OpenCV
0.239 s in MATLAB

Thresholding the crop region takes:
0.0001 s in OpenCV
0.030 s in MATLAB

Border clearing takes:
0.0067 s in skimage
0.0079 s in MATLAB

Closing takes:
0.0009 s in OpenCV
0.0035 s in MATLAB

Hole filling takes:

0.0057 s in MATLAB

Conneced component analysis takes:
0.0004 s in OpenCV
0.0012 s in MATLAB

"""


def cropImageToRegion(image, region):
    return image[region[1]:region[1]+region[3],
                 region[0]:region[0]+region[2]]


class WormImageProcessor:
    """
    Performs image processing tasks needed to find a worm in an image.
    """
    frameRate = 11.5
    expectedWormLength = 1000
    expectedWormWidth = 50
    numberOfPosturePoints = -1
    holeAreaThreshold = 10
    compactnessThreshold = 10
    wormAreaThresholdRange = [0.5, 1.5]

    def __init__(self, pixelSize=0.05, threshold=0.9, backgroundDiskRadius=5,
                 wormDiskRadius=2):
        self.pixelSize = pixelSize
        self.threshold = threshold
        self.backgroundDiskRadius = backgroundDiskRadius
        self.wormDiskRadius = wormDiskRadius

    def autoWormConfiguration(self, wormImage):
        self.wormDiskRadius = round(wormImage.width/2.0*self.pixelSize)

    def applyBackgroundFilter(self, image):
        # Structuring element for bottom hat filtering the background
        bgSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (self.backgroundDiskRadius+1,
                                          self.backgroundDiskRadius+1))
        return 255-cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, bgSE)

    def applyThreshold(self, image):
        junk, thresholded = cv2.threshold(image, 255*self.threshold,
                                          255, cv2.THRESH_BINARY)
        return np.equal(255-thresholded,255)

    def applyMorphologicalCleaning(self, image):
    	"""
    	Applies a variety of morphological operations to improve the detection
    	of worms in the image.
    	Takes 0.030 s on MUSSORGSKY for a typical frame region
    	Takes 0.030 s in MATLAB too
    	"""
        # start with worm == 1
        image = image.copy()
        segmentation.clear_border(image)  # remove objects at edge (worm == 1)
        # fix defects in the thresholding by closing with a worm-width disk
        # worm == 1
        wormSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.wormDiskRadius+1,
                                           	self.wormDiskRadius+1))
        imcl = cv2.morphologyEx(np.uint8(image), cv2.MORPH_CLOSE, wormSE)
        imcl = np.equal(imcl, 1)
        # fix defects by filling holes (filled = 1)
        imholes = np.logical_and(ndimage.binary_fill_holes(image),
                                 np.logical_not(image))
        # combine the two corrections by only filling holes of a certain size
        holes, hierarchy = cv2.findContours(np.uint8(imholes), cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
        # fill holes in inverted closed image if
        # compactness (perimeter^2/area) is above threshold and
        # area is above threshold (This avoids filling an omega turn worm)
        holeAreas = {i: cv2.contourArea(hole) for i, hole
                     in enumerate(holes)}
        holesToFill = [hole for i, hole in enumerate(holes)
                       if holeAreas[i] > self.holeAreaThreshold and
                       cv2.arcLength(hole, True)**2/holeAreas[i] >
                       self.compactnessThreshold]
        imcl = np.uint8(imcl)
        for hole in holesToFill:
            cv2.drawContours(imcl, hole, 0, 1, cv2.cv.CV_FILLED)
        imcl = np.equal(imcl, 1)

        # fix barely touching regions
        # majority with worm pixels == 1 (median filter same?)
        imcl = nf.median_filter(imcl, footprint=[[1, 1, 1],
                                                 [1, 0, 1],
                                                 [1, 1, 1]])
        # diag with worm pixels == 0
        imcl = np.logical_not(bwdiagfill(np.logical_not(imcl)))
        # open with worm pixels == 1
        openSE = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        imcl = cv2.morphologyEx(np.uint8(imcl), cv2.MORPH_OPEN, openSE)
        return np.equal(imcl, 1)

    def identifyPossibleWorms(self, threshImage):
        # do connected component analysis
        contours, hierarchy = cv2.findContours(
            np.uint8(threshImage), cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE)

        # find cc with plausible filled area
        areas = {i: cv2.contourArea(contour)
                 for i, contour
                 in enumerate(contours)}
        n = self.expectedWormAreaPixels()
        l = n*self.wormAreaThresholdRange[0]
        u = n*self.wormAreaThresholdRange[1]
        return [(contour, areas[i]) for i, contour in enumerate(contours)
                if areas[i] > l and areas[i] < u]

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

    def saveConfiguration(self, storeFile, path):
        with h5py.File(storeFile) as f:
            # check whether datasets exist
            if path not in f:
                f.create_group(path)
            g = f[path]
            if 'threshold' not in g:
                g.require_dataset('threshold', (1,), dtype='float64')
                g.require_dataset('backgroundDiskRadius', (1,),
                                  dtype='uint8')
                g.require_dataset('wormDiskRadius', (1,), dtype='uint8')
                g.require_dataset('pixelsPerMicron', (1,), dtype='float64')
                g.require_dataset('holeAreaThreshold', (1,), dtype='uint8')
                g.require_dataset('compactnessThreshold', (1,), dtype='float64')
                g.require_dataset('wormAreaThresholdRange', (2,),
                                  dtype='float64')
                g.require_dataset('frameRate', (1,), dtype='float64')
            # write configuration
            g['threshold'][...] = self.threshold
            g['backgroundDiskRadius'][...] = self.backgroundDiskRadius
            g['wormDiskRadius'][...] = self.wormDiskRadius
            g['pixelsPerMicron'][...] = self.pixelSize
            g['holeAreaThreshold'][...] = self.holeAreaThreshold
            g['compactnessThreshold'][...] = self.compactnessThreshold
            g['wormAreaThresholdRange'][...] = self.wormAreaThresholdRange
            g['frameRate'][...] = self.frameRate


def bwdiagfill(bwimage):
    """ clone of matlab's bwmorph(image,'diag') function? """
    # fills pixels matching the following neighborhoods:
    hoods = [[[0, 1, 0],
              [1, 0, 0],
              [0, 0, 0]],
             [[0, 0, 0],
              [1, 0, 0],
              [0, 1, 0]],
             [[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0]],
             [[0, 1, 0],
              [0, 0, 1],
              [0, 0, 0]]]
    output = bwimage.copy()
    # for each neighborhood, find matching pixels and set them to 1 in the img
    for hood in hoods:
        output = np.logical_or(output,
                               ndimage.binary_hit_or_miss(bwimage, hood))
    return output


def find1Cpixels(bwImage):
    """ identifies 1-connected pixels in image """
    # fills pixels matching the following neighborhoods:
    hoods = [[[1, 0, 0],
              [0, 1, 0],
              [0, 0, 0]],
             [[0, 1, 0],
              [0, 1, 0],
              [0, 0, 0]],
             [[0, 0, 1],
              [0, 1, 0],
              [0, 0, 0]],
             [[0, 0, 0],
              [1, 1, 0],
              [0, 0, 0]],
             [[0, 0, 0],
              [0, 1, 1],
              [0, 0, 0]],
             [[0, 0, 0],
              [0, 1, 0],
              [1, 0, 0]],
             [[0, 0, 0],
              [0, 1, 0],
              [0, 1, 0]],
             [[0, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]]
    output = np.zeros(bwImage.shape, dtype=np.bool)
    # for each neighborhood, find matching pixels and set them to 1 in the img
    for hood in hoods:
        output = np.logical_or(output,
                               ndimage.binary_hit_or_miss(bwImage, hood))
    return output
