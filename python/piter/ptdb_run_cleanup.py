import cPickle
import yaml
import wormtracker as wt
import wormtracker.config as wtc
import wormtracker.parallel as wtp
import wormtracker.wormimageprocessor
import os

wtp.hdf5path = ''
wt.libavPath = ''

#pickleFile = '/home/ptdeboer/worms/videos/n2_day7_short_test_pdb.dat'
configFile = '/home/ptdeboer/worms/worms_config1.yml'

# load pickled WormVideo
#with open(pickleFile, 'rb') as f:
#    wv = cPickle.load(f)

# save WormVideo to YAML configuration file
# the .YAML file is a human-readable configuration format
# yaml.load(f) returns a nested set of lists/dictionaries
#with open(configFile, 'w') as f:
#    wtc.saveWormVideo(wv, f)

# load WormVideo to YAML configuration file
with open(configFile, 'r') as f:
    wvs = wtc.loadWormVideos(f)
    
for video in wvs:
    wtp.cleanUpPostProcess(video)
        