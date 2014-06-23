from fabric.api import *
from fabric.contrib.console import confirm
import yaml
import os
import tempfile
import glob

env.hosts = ['lisa.surfsara.nl']

code_dir = '$HOME/code/WormTracker/python'
work_dir = '~/worms'
video_dir = '~/worms/video'
out_dir = '~/worms/out'


def multistage(configPath):
    configFiles = glob.glob(os.path.join(configPath, '*.yml'))
    for cf in configFiles:
        stage(cf)


def stage(config):
    print 'Loading config file: ' + config
    with open(config, 'r') as f:
        cf = yaml.load(f)
        # change config settings for LISA
        cf['systemSettings']['libavPath'] = ''
        cf['systemSettings']['hdf5Path'] = ''
        for v in cf['videos']:
            sf = v['videoSettings']['storeFile']
            sf = os.path.split(sf)[1]  # remove any directory prefix
            sf = os.path.join(out_dir, sf)  # prepend the output directory
            v['videoSettings']['storeFile'] = sf

            vf = v['videoSettings']['videoFile']
            print 'Copying video file to server: ' + vf
            # copy video to server
            print put(vf, video_dir)
            vf = os.path.split(vf)[1]  # remove any directory prefix
            vf = os.path.join(video_dir, vf)  # prepend the video directory
            print 'Remote video file: ' + vf
            v['videoSettings']['videoFile'] = vf
        configName = os.path.split(config)[1]
        lisaName = os.path.join(tempfile.mkdtemp(), configName)
        with open(lisaName, 'w') as f2:
            yaml.dump(cf, f2)
            print put(lisaName, work_dir)


#def createjob(config):
#    print 'Creating job file: ' + config

