from subprocess import check_output, STDOUT
import sys
import os
import cPickle
import multiprocessing
import time
import h5py
import wormtracker as wt
import wormtracker.postprocess as wtp
from wormtracker import Logger 
# wormtracker.parallel

hdf5path = ''


def batchProcessVideos(wormVideos):
    pool = multiprocessing.Pool()
    results = []
    for video in wormVideos:
        video.saveConfiguration()
        results.append(pool.map_async(processRegion, video.regions))

    for result in results:
        for regionResult in result.get(): 
            Logger.logPrint(regionResult)
    pool.close()
    pool.join()
    Logger.logPrint('Finished analyzing all regions')
    for video in wormVideos:
        cleanUpPostProcess(video)


def parallelProcessRegions(wormVideo):
    wormVideo.saveConfiguration()
    pool = multiprocessing.Pool()
    result = pool.map_async(processRegion, wormVideo.regions)
    Logger.logPrint(','.join([str(r) for r in result.get()]))
    pool.close()
    pool.join()
    Logger.logPrint('Finished analyzing all regions')
    cleanUpProcess(wormVideo)


def processRegion(region):
    Logger.logPrint('Starting analysis of {0} {1}'.format(region.strainName,
                                                region.wormName))
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
        Logger.logPrint('Analysis of {0} {1} took {2} min.'.format(region.strainName,
                                                         region.wormName,
                                                         str(tDuration)))
        return 'Success'
    except(Exception) as e:
        Logger.logPrint('Error during analysis of {0}{1}: {2}'.format(region.strainName,
                                                            region.wormName,
                                                            str(e)))
        return 'Failed'


def cleanUpProcess(wormVideo):
    # merge results into original output file
    # should work
    path, name = os.path.split(wormVideo.storeFile)
    outputFile = os.path.join(path, 'merge_' + name)

    try:
        # copy video info
        Logger.logPrint('Merging HDF5 output files...')
        obj = '/video'

        # Linux: create command, do no escape argument values and keep arguments as separated argument list: 

        # have to copy because the merge will fail if there are any duplicates
        cmd = [hdf5path + 'h5copy', '-i', os.path.join(path, name), '-o',
             outputFile, '-s', obj, '-d', obj, '-p']
        
        Logger.logPrint('Executing:'+' '.join(cmd))
        Logger.logPrint(check_output(cmd))

        # remove premerge file
        os.remove(wormVideo.storeFile)
  
        for region in wormVideo.regions:
            try:
                args = [hdf5path + 'h5copy', '-i', os.path.join(path,'{1}_{2}_{0}'),
                              '-o', os.path.join(path, 'merge_{0}'), '-s',
                              '/worms/{1}/{2}','-d', '/worms/{1}/{2}',
                              '-p']
                #update 
                cmd=[arg.format(name,region.strainName,region.wormName) for arg in args] 

                Logger.logPrint('Executing:'+' '.join(cmd))
                Logger.logPrint('Output:'+check_output(cmd, stderr=STDOUT))
                # remove premerge file
                os.remove(os.path.join(path,
                                       '{1}_{2}_{0}').format(name,
                                                             region.strainName,
                                                             region.wormName))
            except(Exception) as e:
                Logger.logPrint('Error cleaning up:')
                Logger.logPrint('Exception:'+str(e))

        # rename merge file
        os.rename(outputFile, wormVideo.storeFile)
    except(Exception) as e:
        Logger.logPrint('Error cleaning up:')
        Logger.logPrint('Exception:'+str(e))


def parallelPostProcessRegions(storeFile):
    path, name = os.path.split(storeFile)
    outputFile = os.path.join(path, 'merge_' + name)

    try:
        # copy video info
        Logger.logPrint('Splitting HDF5 output files...')
        obj = '/video'

        # Linux: create command, do no escape argument values and keep arguments as separated argument list: 

        # have to copy because the merge will fail if there are any duplicates
        cmd = [hdf5path + 'h5copy', '-i', os.path.join(path, name), '-o',
             outputFile, '-s', obj, '-d', obj, '-p']
        
        Logger.logPrint('Executing:'+' '.join(cmd))
        Logger.logPrint(check_output(cmd))

        postProcessList = []

        # split worms into separate files to avoid IO issues
        with h5py.File(storeFile, 'r') as f:
            strains = f['worms'].keys()
            for strain in strains:
                worms = f['worms'][strain].keys()
                for worm in worms:
                    # split each worm into a separate hdf5 store
                    wormFileName = '{1}_{2}_{0}'.format(name, strain, worm)
                    wormFile = os.path.join(path, wormFileName)
                    # first copy video info over
                    cmd = [hdf5path + 'h5copy', '-i', storeFile, '-o',
                           wormFile, '-s', obj, '-d', obj, '-p']
                    Logger.logPrint('Executing:'+' '.join(cmd))
                    Logger.logPrint(check_output(cmd))

                    # then copy worm data over
                    wormObj = '/worms/{0}/{1}'.format(strain, worm)
                    cmd = [hdf5path + 'h5copy', '-i', storeFile, '-o',
                           wormFile, '-s', wormObj, '-d', wormObj, '-p']
                    Logger.logPrint('Executing:'+' '.join(cmd))
                    Logger.logPrint(check_output(cmd))

                    # add worm to list to postprocess
                    postProcessList.append((wormFile, strain, worm))

        # parallel postprocess each region
        pool = multiprocessing.Pool()
        result = pool.map_async(postProcessRegion, postProcessList)
        Logger.logPrint(','.join([str(r) for r in result.get()]))
        pool.close()
        pool.join()
        Logger.logPrint('Finished analyzing all regions')

        # merge files back together
        for (wormFile, strain, worm) in postProcessList:
            try:
                args = [hdf5path + 'h5copy', '-i', wormFile,
                              '-o', outputFile, '-s',
                              '/worms/{0}/{1}','-d', '/worms/{0}/{1}',
                              '-p']
                #update 
                cmd=[arg.format(strain,worm) for arg in args] 

                Logger.logPrint('Executing:'+' '.join(cmd))
                Logger.logPrint('Output:'+check_output(cmd, stderr=STDOUT))
                
                # remove premerge file
                os.remove(wormFile)
            except(Exception) as e:
                Logger.logPrint('Error cleaning up:')
                Logger.logPrint('Exception:'+str(e))

        # remove original file
        os.remove(storeFile)

        # rename merge file
        os.rename(outputFile, storeFile)
    except(Exception) as e:
        Logger.logPrint('Error cleaning up:')
        Logger.logPrint('Exception:'+str(e))


def postProcessRegion(jobInfo):
    wormFile, strain, worm = jobInfo
    with h5py.File(wormFile, 'r+') as f:
        pp = wtp.WormTrajectoryPostProcessor(f, strain, worm)
        pp.postProcess()
        pp.store()


if __name__ == '__main__':
    multiprocessing.freeze_support()
