import sys
import cv2
from skimage import segmentation
from roitools import *
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import wormtracker as wt
import wormimageprocessor as wp

class WormVideo:
    imageProcessor = wp.WormImageProcessor()
    firstFrame = None
    pixelsPerMicron = None
    regions = []
    frameSize = None

    def __init__(self, videoFile, storeFile='temp.h5',
                 videoInfoStorePath='/video', 
                 resultsStorePath='/worms',
                 numberOfRegions=16, allSameStrain=True,
                 referenceDistance=25000):
        self.videoFile = videoFile
        self.numberOfRegions = numberOfRegions
        self.allSameStrain = allSameStrain
        self.referenceDistance = referenceDistance
        self.storeFile = storeFile
        self.videoInfoStorePath = videoInfoStorePath
        self.resultsStorePath = resultsStorePath

    def readFirstFrame(self):
        video = cv2.VideoCapture()
        if video.open(self.videoFile):
            success, firstFrame = video.read()
            if not success:
                raise Exception("Couldn't read video")
            else:
                firstFrameChannels = cv2.split(firstFrame)
                self.firstFrame = firstFrameChannels[0]
                self.frameSize = self.firstFrame.shape
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
                         region[1] + region[2])

        for i in xrange(self.numberOfRegions):
            raw_input("Select region " + str(i) + " on the figure...")
            drawAllRegions()  # show all regions already picked
            sel = RectangleRegionSelector()  # track selection
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
            self.addRegion(tuple([round(x) for x in region[0]]),
                           region[1], region[2])

    def addRegion(self, regionBounds, strain, name):
        """Adds the video region containing one worm.

        regionBounds: (x,y,w,h)
        strain: The strain name
        name: The worm identifier
        """
        wr = wt.WormVideoRegion(self.videoFile, self.imageProcessor,
                                self.storeFile,
                                regionBounds, self.pixelsPerMicron,
                                outputPrefix='temp',
                                resultsStorePath=self.resultsStorePath,
                                strainName=strain,
                                wormName=name)
        self.regions.append(wr)

    def determinePixelSize(self):
        if self.firstFrame is None:
            self.readFirstFrame()
        raw_input("Draw a " + str(self.referenceDistance) +
                  " um line on the figure...")
        plt.imshow(self.firstFrame, cmap=plt.gray())
        sel = LineRegionSelector()
        plt.show()
        self.imageProcessor.pixelSize = (sel.distance() /
                                         self.referenceDistance)
        print ("The pixel size is " + str(1.0/self.imageProcessor.pixelSize) +
               " um/px.")

    def defineFoodRegions(self):
        if self.firstFrame is None:
            self.readFirstFrame()
        raw_input("Draw a circle covering the food for each region...")
        for region in self.regions:
            crop = region.cropRegion
            plt.imshow(wt.cropImageToRegion(self.firstFrame, crop),
                       plt.gray())
            plt.title(region.strainName + " " + region.wormName)
            sel = CircleRegionSelector()
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

    def testWormIdentification(self):
        raise NotImplemented()

    def saveConfiguration(self):
        with h5py.File(self.resultsStoreFile) as f:
            pre = self.videoInfoStorePath
            # check whether datasets exist
            if pre not in f:
                g = f.create_group(pre)
                dt = h5py.special_dtype(vlen=str)
                g.create_dataset('videoFile', (1, 255), dtype=dt)

            # write configuration
            g = f[pre]
            # strip directory info from file
            path, fileName = os.split(self.videoFile)
            g['videoFile'] = fileName
            # save imaging configuration
            self.imageProcessor.saveConfiguration(self.resultsStoreFile,
                                                  self.videoInfoStorePath)
            # save information for each region
            for region in self.regions:
                region.saveConfiguration()


def main(argv):
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="the input video file")
    #parser.add_argument('-o', '--output', help="the output image file")
    args = parser.parse_args()
    wv = WormVideo(args.input)
    wv.determinePixelSize()
    wv.defineRegions()
    wv.defineFoodRegions()
    wv.testBackgroundFilter()
    wv.testThreshold()
    wv.testWormIdentification()


if __name__ == "__main__":
    main(sys.argv[1:])

#  import os
#  os.chdir('D:\\Stephen\\Documents\\Code\\wormtracker-matlab')
#  videoFile = 'D:\\test_segment_unc_gray.avi'
