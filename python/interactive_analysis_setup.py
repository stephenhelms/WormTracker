import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import wormtracker as wt
import wormimageprocessor as wp

# not using this for now
def main(argv):
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="the input video file")
    #parser.add_argument('-o', '--output', help="the output image file")
    args = parser.parse_args()
    wv = wt.WormVideo(args.input)
    wv.determinePixelSize()
    wv.defineRegions()
    wv.defineFoodRegions()
    wv.testBackgroundFilter()
    wv.testThreshold()
    wv.testWormIdentification()


if __name__ == "__main__":
    #main(sys.argv[1:])
    import Tkinter, tkFileDialog

    root = Tkinter.Tk()
    root.withdraw()

    print 'Select the video file...'
    videoFile = tkFileDialog.askopenfilename()
    
    print 'Select the output data file...'
    storeFile = tkFileDialog.asksaveasfilename(defaultextension='h5')

    nRegions = 16
    wv = wt.WormVideo(videoFile, storeFile=storeFile,
                      numberOfRegions=nRegions,
                      allSameStrain=True)
    wv.determinePixelSize()
    wv.defineRegions()
    wv.defineFoodRegions()

    wv.testBackgroundFilter()
    wv.testThreshold()
    wv.testWormIdentification()
    # if any of these are problematic, adjust
    # wv.imageProcessor properties

    wv.processRegions()


#  import os
#  os.chdir('D:\\Stephen\\Documents\\Code\\wormtracker-matlab')
#  videoFile = 'D:\\test_segment_unc_gray.avi'
#  import cPickle
#  with open('wormImageData.dat','rb') as f:
#    wormImageData = cPickle.load(f)
#  reload(wt)
#  wim = wt.WormImage(wv.regions[0], gray, cl, worms[0][0])
#  wim.measureWorm()
#  wim.store('test.h5','/test',1)
