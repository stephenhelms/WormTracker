"""
Runs a batch analysis of worm videos from a YAML configuration file
generated using the wormtracker.config module.

Usage:
python runbatchanalysis.py -i configfile.yml

The data files will be output in the folder where the store file is located.
After parallel processing, the final file has merge_ prepended to the name.
Another set of files prepended with {strain}_{wormname} are generated but not
automatically deleted in case something goes wrong.
"""

import sys
import argparse
import cPickle
import yaml
import time 

# Change this to the directory where the code is stored
import wormtracker as wt
import wormtracker.config as wtc
import wormtracker.parallel as wtp
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
                        help="the input YAML configuration file")
    parser.add_argument('-s', '--serial',
                        help="run in serial mode",
                        action="store_true")
    args = parser.parse_args()

    batchStart = time.time()

    Logger.logPrint("Starting batchprocess: "+args.input)
    Logger.logPrint("Start time:"+timeStr(batchStart))

    # load WormVideo to YAML configuration file
    with open(args.input, 'r') as f:
        wvs = wtc.loadWormVideos(f)

    # run analysis on regions
    if args.serial:
        for wv in wvs:
            wv.processRegions()
    else:
        wtp.batchProcessVideos(wvs)
    batchStop = time.time()
    Logger.logPrint("Start time:"+timeStr(batchStart))
    Logger.logPrint("End time  :"+timeStr(batchStop))
    Logger.logPrint("Total time:"+timeStr(batchStop-batchStart))

    return 'Success'


if __name__ == "__main__":
    main(sys.argv)
