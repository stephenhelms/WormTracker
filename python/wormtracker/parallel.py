import sys, os, cPickle, multiprocessing, time
from subprocess import check_output
import wormtracker as wt

# wormtracker.parallel

hdf5path = 'C:\\hdf5\\'

def parallelProcessRegions(wormVideo):
    wormVideo.saveConfiguration()
    pool = multiprocessing.Pool()
    result = pool.map_async(processRegion, wormVideo.regions)
    print result.get()
    pool.close()
    pool.join()
    print 'Finished analyzing all regions'


def processRegion(region):
    print 'Starting analysis of {0} {1}'.format(region.strainName,
                                                region.wormName)
    originalOutput = region.resultsStoreFile
    # split output to a different file
    path, name = os.path.split(region.resultsStoreFile)
    newName = (region.strainName+'_'+region.wormName+'_'+name)
    region.resultsStoreFile = os.path.join(path, newName)
    tStart = time.clock()
    region.saveConfiguration()
    region.process()
    tFinish = time.clock()
    tDuration = (tFinish - tStart) / 60
    print 'Analysis of {0} {1} took {2} min.'.format(region.strainName,
                                                     region.wormName,
                                                     str(tDuration))
    # merge results into original output file
    # should work
    path, name = os.path.split(wormVideo.storeFile)
    outputFile = os.path.join(path, 'merge_' + name)

    # copy video info
    print 'Merging HDF5 output files...'
    obj = '\"/video\"'
    c = [hdf5path + 'h5copy', '-i', os.path.join(path, name), '-o',
         outputFile, '-s', obj, '-d', obj, '-p']
    print check_output(' '.join(c))

    # copy region files
    cmds = [' '.join([hdf5path + 'h5copy', '-i', os.path.join(path,
                                                              '{1}_{2}_{0}'),
                     '-o', os.path.join(path, 'merge_{0}'), '-s',
                     '\"/worms/{1}/{2}\"','-d', '\"/worms/{1}/{2}\"',
                     '-p']).format(name,
                                   region.strainName,
                                   region.wormName)
            for region in wormVideo.regions]
    for c in cmds:
        print check_output(c)
