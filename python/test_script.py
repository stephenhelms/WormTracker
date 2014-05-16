import os
import cPickle
# Change this to the directory where the code is stored
os.chdir('D://Stephen/Documents/Code/wormtracker-matlab/python')
import wormtracker as wt
import wormtracker.parallel as wtp

"""
This script runs a performance test for the serial analysis on a
single region of a video segment.

Performance results on MUSSORGSKY:
Intel Core i7-2600K @ 3.4 GHz (8 threads)
8 GB RAM
Windows 7
64-bit Anaconda Python 2.7 environment

Average time per frame was 0.074214055519 s.
Projected time for 1 hour of video for a single region: 49.4760370126 min.
Projected time for 16 regions: 13.19360987 h.
"""

# configure libav
# wt.libavPath = C:\\libav\\bin

# video settings
# change this to the video folder
videoFile = 'D:\\n2_day7_short_test.avi'
storeFile = 'D:\\test.h5'
pickleFile = 'D:\\N2_a_b_day_7.dat'

# load object
with open(pickleFile, 'rb') as f:
	wv = cPickle.load(f)

# only analyze 1 region
wv.regions = wv.regions[:1]

# update references
wv.updateVideoFile(videoFile)
wv.storeFile = storeFile
for region in wv.regions:
	region.resultsStoreFile = storeFile

# run analysis
import time
tStart = time.clock()
wv.processRegions()  # analyzes each region serially
tStop = time.clock()
tDuration = tStop - tStart
print 'Processing in serial took {0} min.'.format(str(tDuration/60))
tPerFrame = tDuration / wv.nFrames
print 'Average time per frame was {0} s.'.format(str(tPerFrame))
tPerRegion = tPerFrame*40000
print ('Projected time for 1 hour of video for a single region: ' +
       '{0} min.'.format(str(tPerRegion/60)))
tVideo = tPerRegion*16
print ('Projected time for 16 regions: {0} h.'.format(str(tVideo/60**2)))
