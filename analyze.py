import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import h5py
from subprocess import call

#Interactive script — replace VideoReader with the Python/OpenCV approach (only need a single frame anyway)
#Automated analysis, parallelized per worm (16/video in the new ones):

class WormVideoRegion:
	backgroundDiskSize = 5
	backgroundThreshold = 0.9
	wormDiskSize = 2
	expectedWormLength = 1000
	expectedWormWidth = 50

	def __init__(self, videoFile, outputPrefix = 'temp', cropRegion, pixelSize, strainName = 'Unknown', wormName = ''):
		self.videoFile = videoFile
		self.outputPrefix = outputPrefix
		self.cropRegion = cropRegion

		self.pixelSize = pixelSize
		px = expectedWormLengthPixels()
		if px > 60:
			self.numberOfPosturePoints = 50
		elif px > 40:
			self.numberOfPosturePoints = 30
		elif px > 20:
			self.numberOfPosturePoints = 15
		else: # not enough points to do postural analysis
			self.numberOfPosturePoints = 0

		self.strainName = strainName
		self.wormName = wormName

	def expectedWormLengthPixels(self):
		""" Returns the expected length of a worm in pixels """
		return self.expectedWormLength*self.pixelSize

	def expectedWormWidthPixels(self):
		""" Returns the expected width of a worm in pixels """
		return self.expectedWormWidth*self.pixelSize

	def expectedWormAreaPixels(self):
		""" Returns the expected area of a worm in pixels^2 """
		return expectedWormLengthPixels()*expectedWormWidthPixels()

	def process(self):
		generateCroppedFilteredVideo()
		generateThresholdedVideo()
		identifyWorm()
		outlineWorm()
		skeletonizeWorm()
		measureWorm()

	def generateCroppedFilteredVideo(self):
		""" Uses libav to crop and apply a bottom hat filter to the video """
		self.croppedFilteredVideoFile = self.outputPrefix + '_cropped.avi'
		
		call('avconv','-i',self.videoFile,'-vf','crop='+cropRegionForAvconv(self), '-c:v rawvideo',
			'-pix_fmt yuv420p',self.croppedFilteredVideoFile)

	def generateThresholdedVideo(self):
		""" Uses libav to threshold a video """
		self.thresholdedVideoFile = self.outputPrefix + '_thresholded.avi'

		call('avconv','-i',self.croppedFilteredVideoFile,'-vf','?') # need to figure out how to do this

	def cropRegionForAvconv(self):
		return self.cropRegion[0] + ':' + self.cropRegion[1] + ':' + self.cropRegion[2] + ':' + 
			self.cropRegion[3]

	def identifyWorm(self):
		# loop through video frames
		# find worm (largest connected component with plausible filled area)
		# crop frame to worm
		# store in HDF5 container
		return None

	def outlineWorm(self):
		return None

	def skeletonizeWorm(self):
		return None

	def measureWorm(self):
		# loop through video frames
		# measure centroid
		# measure width
		# measure length
		# measure body angles
		# (store everything in HDF5)