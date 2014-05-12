# import worm tracker package
import os
os.chdir('//system-biologysrv.amolf.nl/users/Helms/Code/wormtracker-matlab')
import wormtracker as wt

# configure libav
# wt.libavPath = C:\\libav\\bin

# video settings
os.chdir('//system-biologysrv.amolf.nl/users/Koers/worm_videos/day_7')
videoFile = '2014-04-14_n2_a_b_day_7.avi'
storeFile = '2014-04-14_n2_a_b_day_7_processed.h5'

wv = wt.WormVideo(videoFile, storeFile=storeFile,
                  allSameStrain=True)

# configure video
# draw a line across the width of one of the two slides
wv.determinePixelSize()
# draw a box around each region, avoiding dark regions if possible
# (can change # of regions by setting wv.numberOfRegions)
wv.defineRegions()
# draw a circle around the food region
wv.defineFoodRegions()

# test video settings
wv.testBackgroundFilter()
wv.testThreshold()
# was crashing on not finding the worm, maybe fixed now?
wv.testWormIdentification()

# process video
wv.saveConfiguration()
wv.processRegions()  # analyzes each region serially