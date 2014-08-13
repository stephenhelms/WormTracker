"""
Runs a batch post processing of worm videos from a YAML configuration file
generated using the wormtracker.config module or from the output .h5 file

Usage:
python runbatchanalysis.py -i configfile.yml
or
python runbatchanalysis.py -i output.h5
"""

import os
import sys
import argparse
import yaml
import time 

# Change this to the directory where the code is stored
import wormtracker.config as wtc
import wormtracker.parallel as wtp
import wormtracker.postprocess as wtpp
import h5py
from wormtracker import Logger

def timeStr(time):
    ss=(time%60)
    time=time/60 
    mm=(time%60)
    time=time/60
    hh=time%24; 
    return "{:0=2.0f}:{:0=2.0f}:{:0=2.0f}".format(hh,mm,ss); 

def main(argv):
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help="the input YAML configuration file or h5 store file")
    parser.add_argument('-s', '--serial',
                        help="run in serial mode",
                        action="store_true")
    args = parser.parse_args()

    batchStart = time.time()

    Logger.logPrint("Starting batchprocess: "+args.input)
    Logger.logPrint("Start time:"+timeStr(batchStart))

    fileName, fileExtension = os.path.splitext(args.input)
    # If given a store file (.h5), just use that
    if fileExtension == '.h5':
        storeFiles = [args.input]
    else:
        with open(args.input, 'r') as f:
            storeFiles = wtc.extractStoreFileList(f)

    # postprocessing
    if args.serial:
        for storeFile in storeFiles:
            with h5py.File(wv.storeFile, 'r+') as f:
                strains = f['worms'].keys()
                for strain in strains:
                    worms = f['worms'][strain].keys()
                    for worm in worms:
                        pp = wtpp.WormTrajectoryPostProcessor(f, strain, worm)
                        pp.postProcess()
                        pp.store()
    else:
        for storeFile in storeFiles:
            Logger.logPrint('Post-processing: ' + storeFile)
            wtp.parallelPostProcessRegions(storeFile)

    batchStop = time.time()
    Logger.logPrint("Start time:"+timeStr(batchStart))
    Logger.logPrint("End time  :"+timeStr(batchStop))
    Logger.logPrint("Total time:"+timeStr(batchStop-batchStart))

    return 'Success'


if __name__ == "__main__":
    main(sys.argv)
