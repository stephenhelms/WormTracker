from subprocess import check_output
import sys
import os
import cPickle
import multiprocessing
import time
import wormtracker as wt

# wormtracker.parallel

hdf5path = 'C:\\hdf5\\'


def batchProcessVideos(wormVideos):
    pool = multiprocessing.Pool()
    results = []
    for video in wormVideos:
        video.saveConfiguration()
        results.append(pool.map_async(processRegion, video.regions))

    for result in results:
        print result.get()
    pool.close()
    pool.join()
    print 'Finished analyzing all regions'
    for video in wormVideos:
        cleanUpPostProcess(video)


def parallelProcessRegions(wormVideo):
    wormVideo.saveConfiguration()
    pool = multiprocessing.Pool()
    result = pool.map_async(processRegion, wormVideo.regions)
    print result.get()
    pool.close()
    pool.join()
    print 'Finished analyzing all regions'
    cleanUpPostProcess(wormVideo)


def processRegion(region):
    print 'Starting analysis of {0} {1}'.format(region.strainName,
                                                region.wormName)
    try:
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
        return 'Success'
    except(Exception) as e:
        print 'Error during analysis of {0}{1}: {2}'.format(region.strainName,
                                                            region.wormName,
                                                            str(e))
        return 'Failed'


def cleanUpPostProcess(wormVideo):
    # merge results into original output file
    # should work
    path, name = os.path.split(wormVideo.storeFile)
    outputFile = os.path.join(path, 'merge_' + name)

    try:
        # copy video info
        print 'Merging HDF5 output files...'
        obj = '/video'

        # Linux: create command, do no escape argument values and keep arguments as separated argument list: 

        cmd = [hdf5path + 'h5copy', '-i', os.path.join(path, name), '-o',
             outputFile, '-s', obj, '-d', obj, '-p']
        
        print 'Executing:',' '.join(cmd)
        print check_output(cmd);
         
        for region in wormVideo.regions:
            
            args = [hdf5path + 'h5copy', '-i', os.path.join(path,'{1}_{2}_{0}'),
                          '-o', os.path.join(path, 'merge_{0}'), '-s',
                          '/worms/{1}/{2}','-d', '/worms/{1}/{2}',
                          '-p']
            #update 
            cmd=[arg.format(name,region.strainName,region.wormName) for arg in args] 

            print 'Executing:',' '.join(cmd)
            print check_output(cmd)
        
    except(Exception) as e:
        print 'Error cleaning up:'
        print e


if __name__ == '__main__':
    multiprocessing.freeze_support()
