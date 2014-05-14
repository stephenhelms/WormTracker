import sys, os, cPickle, multiprocessing, time
import wormtracker

# wormtracker.parallel


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
    # TODO - h5merge
