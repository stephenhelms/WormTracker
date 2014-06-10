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
# Change this to the directory where the code is stored
import wormtracker as wt
import wormtracker.config as wtc
import wormtracker.parallel as wtp


def main(argv):
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help="the input YAML configuration file")
    args = parser.parse_args()

    # load WormVideo to YAML configuration file
    with open(args.input, 'r') as f:
        wvs = wtc.loadWormVideos(f)

    # run analysis on region
    wtp.batchProcessVideos(wvs)

    return 'Success'


if __name__ == "__main__":
    main(sys.argv)
