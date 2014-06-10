import h5py
import os
import yaml
import wormtracker as wt
import wormtracker.parallel as wtp
import wormtracker.postprocess as wtpp

def saveWormVideo(video, f):
    systemSettings = getSystemConfigDict()
    videoSettings = getVideoConfigDict(video)
    regionSettings = [getRegionConfigDict(region)
                      for region in video.regions]
    yaml.dump({'systemSettings': systemSettings,
               'videos': [{
                    'videoSettings': videoSettings,
                    'regions': regionSettings
                }]
            }, f)


def getSystemConfigDict():
    return {
        'libavPath': wt.libavPath,
        'hdf5Path': wtp.hdf5path
    }


def getVideoConfigDict(video):
    return {
        'videoFile': video.videoFile,
        'storeFile': video.storeFile,
        'backgroundDiskRadius': video.imageProcessor.backgroundDiskRadius,
        'pixelsPerMicron': video.imageProcessor.pixelSize,
        'threshold': video.imageProcessor.threshold,
        'wormAreaThresholdRange': video.imageProcessor.wormAreaThresholdRange,
        'wormDiskRadius': video.imageProcessor.wormDiskRadius,
        'expectedWormLength': video.imageProcessor.expectedWormLength,
        'expectedWormWidth': video.imageProcessor.expectedWormWidth,
        'frameRate': video.imageProcessor.frameRate,
        'numberOfPosturePoints': video.imageProcessor.numberOfPosturePoints
    }


def getRegionConfigDict(region):
    if type(region.foodCircle[0]) is not float:
            region.foodCircle = tuple(float(n) for n in region.foodCircle)
    return {
        'strainName': region.strainName,
        'wormName': region.wormName,
        'cropRegion': region.cropRegion,
        'foodCircle': region.foodCircle
    }


def loadWormVideos(f):
    config = yaml.load(f)
    # load system settings
    wt.libavPath = config['systemSettings']['libavPath']
    wtp.hdf5path = config['systemSettings']['hdf5Path']

    videos = []
    for videoC in config['videos']:
        # load video object and configure
        vs = videoC['videoSettings']
        video = wt.WormVideo(vs['videoFile'],
                             vs['storeFile'])
        video.imageProcessor.backgroundDiskRadius = vs['backgroundDiskRadius']
        video.imageProcessor.pixelSize = vs['pixelsPerMicron']
        video.imageProcessor.threshold = vs['threshold']
        video.imageProcessor.wormAreaThresholdRange = vs['wormAreaThresholdRange']
        video.imageProcessor.wormDiskRadius = vs['wormDiskRadius']
        video.imageProcessor.expectedWormLength = vs['expectedWormLength']
        video.imageProcessor.expectedWormWidth = vs['expectedWormWidth']
        video.imageProcessor.frameRate = vs['frameRate']
        video.imageProcessor.numberOfPosturePoints = vs['numberOfPosturePoints']

        # create regions
        for region in videoC['regions']:
            wr = video.addRegion(region['cropRegion'],
                                 region['strainName'],
                                 region['wormName'])
            wr.foodCircle = region['foodCircle']

        # re-initialize state
        video.readFirstFrame(askForFrameRate=False)
        video.getNumberOfFrames()
        videos.append(video)
    return videos
