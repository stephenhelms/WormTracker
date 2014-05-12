import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir('D:\\Stephen\\Documents\\Code\\wormtracker-matlab')
videoFile = 'D:\\test_segment_unc_gray.avi'
import cPickle
with open('wormImageData.dat','rb') as f:
    wormImageData = cPickle.load(f)

import wormtracker as wt

reload(wt)
wim = wt.WormImage(wormImageData['region'],
	               wormImageData['filtered'],
	               wormImageData['cl'],
	               wormImageData['worm'])
wim.measureWorm()
wim.plot()
plt.show()

reload(wt)
wim = wt.WormImage(wormImageData['region'],
	               wormImageData['filtered'],
	               wormImageData['cl'],
	               wormImageData['worm'])
wim.measureWorm()
wim.store('test.h5', '/test', 1)

##
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('D:\\Stephen\\Documents\\Code\\wormtracker-matlab')
videoFile = 'D:\\test_segment_unc_gray.avi'

import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir('D:\\Stephen\\Documents\\Code\\wormtracker-matlab')
videoFile = 'D:\\test_segment_unc_gray.avi'
import wormtracker as wt
wv = wt.WormVideo(videoFile)
wv.determinePixelSize()
wv.numberOfRegions = 1
wv.defineRegions()
wv.defineFoodRegions()

#
for region in wv.regions:
	region.process()

#
wv.testBackgroundFilter()
wv.testThreshold()
wv.testWormIdentification()
wv.saveConfiguration()
