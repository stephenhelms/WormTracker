import argparse
import sys
import os
import cPickle
import wormtracker as wt
import wormtracker.parallel as wtp
import glob

"""
This is designed to be used as a command-line method for analyzing
a batch of a WormVideo objects in a separate process.
"""


def possibleWormVideoFiles(directory=None, extension='.dat'):
    if directory is None:
        # use current directory
        directory = os.curdir
    return glob.glob(os.join(directory, '*'+extension))


def main(argv):
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='*',
                        help="the input pickled WormVideo object(s)")
    args = parser.parse_args()

    if 'input' in args:
        inputFiles = parser.input
        if len(inputFiles) == 1:
            # check whether a directory was passed
            path, name = os.path.split(inputFiles)
            if name is None:
                inputFiles = possibleWormVideoFiles(path)
    else:
        # grab all .dat files from folder
        inputFiles = possibleWormVideoFiles()

    wormVideos = []
    for inputFile in inputFiles:
        print 'Loading ' + inputFile
        # load WormVideo object
        try:
            with open(parser.input, 'rb') as f:
                wormVideos.append(cPickle.load(f))
        except IOError as e:
            s = "I/O error({0}) while unpickling WormVideo: {1}"
            print s.format(e.errno, e.strerror)
            raise
        except Exception as e:
            # can't do anything, just report error and exit gracefully
            print ('Failed to open or unpickle WormVideo object.' +
                   ' Received error: ' + sys.exc_info()[0])

    # run analysis on all worm videos
    wtp.batchProcessVideos(wormVideos)

    return 'Success'


if __name__ == "__main__":
    main(sys.argv)
