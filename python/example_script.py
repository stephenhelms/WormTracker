# import worm tracker package
import os
# Change this to the directory where the code is stored
os.chdir('D://Documents/Stephen/Code/wormtracker-matlab')
import wormtracker as wt

# configure libav
# wt.libavPath = C:\\libav\\bin

# video settings
# change this to the video folder
os.chdir('//system-biologysrv.amolf.nl/users/Koers/worm_videos/day_7')
videoFile = '2014-04-14_n2_a_b_day_7.avi'
storeFile = '2014-04-14_n2_a_b_day_7_processed.h5'
pickleFile = '2014-04-14_n2_a_b_day_7.dat'

# if all are not from the same strain, change to allSameStrain=False
# if number of regions < 16, add numberOfRegions=N
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
wv.testWormIdentification()

# process video
wv.saveConfiguration()

# save object to pickle file
import cPickle
with open(pickleFile, 'wb') as f:
    cPickle.dump(wv, f)


#
# DONE LATER
#
import os
# Change this to the directory where the code is stored
os.chdir('D://Documents/Stephen/Code/wormtracker-matlab')
import wormtracker as wt
# load object
import cPickle
pickleFile = '2014-04-14_n2_a_b_day_7.dat'
with open(pickleFile, 'rb') as f:
    wv = cPickle.load(f)

# run analysis
wv.processRegions()  # analyzes each region serially