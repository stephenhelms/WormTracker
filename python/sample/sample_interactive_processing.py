import os
# change this to your code path
os.chdir('/Users/stephenhelms/Code/WormTracker/python')

"""
The config file specifies all the parameters for analyzing a video.

NOTE:
In the config file, edit the hdf5path
if hdf5 tools are not on your system path
(h5copy is called at the end of parallel analysis)

Also, edit the storeFile and videoFile paths at the end
The directory for the store file needs to already exist
"""
configFile = './sample/short_test.yml'

# load the WormVideo object using the config module
import wormtracker.config as wtc
with open(configFile, 'r') as f:
    wvs = wtc.loadWormVideos(f)
# a config file can specify multiple videos,
# but here there's only one
wv = wvs[0]

# run the analysis in serial
wv.processRegions()

# run postprocessing
import h5py
import wormtracker.postprocess as wtp
with h5py.File(wv.storeFile, 'r+') as f:
	strains = f['worms'].keys()
	for strain in strains:
		worms = f['worms'][strain].keys()
		for worm in worms:
			pp = wtp.WormTrajectoryPostProcessor(f, strain, worm)
			pp.postProcess()
			pp.store()
