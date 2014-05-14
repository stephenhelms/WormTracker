import argparse, sys, cPickle, multiprocessing
import wormtracker as wt
import wormtracker.parallel as wtp

"""
This is designed to be used as a command-line method for analyzing
a region of a WormVideo object in a separate process.
"""

def main(argv):
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
    	                help="the input pickled WormVideo object")
    parser.add_argument('-r', '--region',
    	                help="the region to analyze")
    args = parser.parse_args()
    
    # grab region number
    regionNumber = args.region

    # load WormVideo object
    try:
    	with open(parser.input, 'rb') as f:
    		wv = cPickle.load(f)
    except IOError as e:
    	s = "I/O error({0}) while unpickling WormVideo: {1}"
        print s.format(e.errno, e.strerror)
        raise
    except Exception as e:
    	# can't do anything, just report error and exit gracefully
        print ('Failed to open or unpickle WormVideo object.' +
        	   ' Received error: ' + sys.exc_info()[0])
        raise

    # run analysis on region
    wtp.processRegion(wv.regions[regionNumber])

    return 'Success'


if __name__ == "__main__":
    main(sys.argv)