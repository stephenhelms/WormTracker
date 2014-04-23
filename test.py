import numpy as np
import matplotlib.pyplot as plt

import wormtracker as wt

import os
os.chdir('D:\\Stephen\\Documents\\Code\\wormtracker-matlab')
videoFile = 'D:\\test_segment_unc_gray.avi'
import cPickle
with open('wormImageData.dat','rb') as f:
    wormImageData = cPickle.load(f)
    
reload(wt)
wim = wt.WormImage(wormImageData['region'],
	               wormImageData['filtered'],
	               wormImageData['cl'],
	               wormImageData['worm'])
wim.measureWorm()
wim.plot()
plt.show()

wim.store('test.h5', '/test', 1)
