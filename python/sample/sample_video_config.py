import os
# change this to your code path
os.chdir('/Users/stephenhelms/Code/WormTracker/python')

# change this to point to your video file
videoFile = '/Users/stephenhelms/Dropbox/OpenData/short_test.avi'
# this is where the results will be stored (can change)
storeFile = './out/short_test.h5'

# load wormtracker
import wormtracker as wt
# create wormvideo object with 1 region (16 actually on video)
# and 25 mm reference distance (the width of one slide)
wv = wt.WormVideo(videoFile, storeFile, numberOfRegions=1, referenceDistance=25000)

# 1) determine pixel size
# this will first ask you for the frame rate (11.5)
# you then press enter and it will show you the first frame
# you need to draw a line across the width of one of the slides by holding the mouse
# it should report a pixel size of ~22 um/px
wv.determinePixelSize()

# 2) define regions
# this will ask you to draw a box for each analysis region containing 1 worm
# it will ask you to enter a strain name (N2)
# and optionally a name (if you just press enter, it will use consecutive numbers)
wv.defineRegions()

# 3) define food regions (if present)
# this will ask you to draw a circle covering the food in each region
# hold down the mouse on the center of the food and drag outward to covering
wv.defineFoodRegions()

# 4) enter expected worm length and width or manually set imaging parameters
wv.imageProcessor.expectedWormWidth = 70
wv.imageProcessor.expectedWormLength = 1000
wv.imageProcessor.determineNumberOfPosturePoints()  # or set manually

# 5) test imaging conditions and tune if necessary
# A: background filtering
# parameters controlling this:
# wv.imageProcessor.backgroundDiskRadius
wv.testBackgroundFilter()
# B: thresholding
# wv.imageProcessor.threshold
wv.testThreshold()
# C: morphological cleaning (region number)
# wv.imageProcessor.wormDiskRadius (should be ~ half width of worm in pixels)
wv.testMorphologicalCleaning(0)
# D: worm identification
# wv.imageProcessor.expectedWormWidth
# wv.imageProcessor.expectedWormLength
# wv.imageProcessor.wormAreaThresholdRange
wv.testWormIdentification()

# 6) save configuration to a .yml config file
import wormtracker.config as wtc
configFile = 'test.yml'
with open(configFile, 'w') as f:
    wtc.saveWormVideo(wv, f)
