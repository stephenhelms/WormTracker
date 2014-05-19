import h5py
import os
os.chdir('D:\\Stephen\\Documents\\Code\\wormtracker-matlab\\python')
import wormtracker.postprocess as pp


# configuration
results = 'D:\\2014-04-14_n2_a_b_day_7_processed.h5'
f = h5py.File(results, 'r+')  # warning: read-write permission!

# postprocess all trajectories
for strain in f['worms'].keys():
	for wormID in f['worms'][strain].keys():
		print 'Post-processing {0} {1}'.format(strain, wormID)
		wtpp = pp.WormTrajectoryPostProcessor(f, strain, wormID)
		wtpp.postProcess()  # this can take a while...
		wtpp.store()  # store results

f.close()