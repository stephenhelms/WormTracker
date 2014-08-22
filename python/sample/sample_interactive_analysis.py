import os
# change this to your code path
os.chdir('/Users/stephenhelms/Code/WormTracker/python')

# use the config file to find the results
# can also manually set storeFile to the .h5 file
configFile = './sample/short_test.yml'
import wormtracker.config as wtc
with open(configFile, 'r') as f:
    wvs = wtc.loadWormVideos(f)
# a config file can specify multiple videos,
# but here there's only one
wv = wvs[0]
storeFile = wv.storeFile

# manual load .h5 file
storeFile = '/Users/stephenhelms/Dropbox/OpenData/short_test.h5'

# load results
import h5py
import wormtracker.analysis as wta
wta.configureMatplotLibStyle()  # make prettier plots
f = h5py.File(storeFile, 'r')  # data is only loaded in memory as needed (good for handling large ensembles)
strain = 'N2'

# INDIVIDUAL TRAJECTORIES
worm = '1'
traj = wta.WormTrajectory(f, strain, worm)
traj.plotTrajectory()

# show movie
import matplotlib.pyplot as plt
import wormtracker.visualize as wtv
aniTraj = wtv.AnimatedWormTrajectoryWithImage(traj)
aniTraj.getAnimation()
plt.show()

# plot speed time series
traj.plotSpeed()

# plot speed distribution
traj.plotSpeedDistribution()

# plot mean-squared displacmeent
# (statistical description of position over time)
# this function is currently slow...
traj.plotMeanSquaredDisplacement()

# plot speed autocorrelation
# (it's noisy in this case because the video is short)
traj.plotSpeedAutocorrelation()

# plot body bearing over time
# there is a linear trend that needs to be fit (not implemented)
traj.plotBodyBearing()

# plot body bearing autocorrelation
# this needs to have the linear trend subtracted (not implemented)
traj.plotBodyBearingAutocorrelation()

# plot reversal state autocorrelation
# should exponentially decay with a 1-2 s time constant
# related to the reversal duration
traj.plotReversalStateAutocorrelation()

# there are also commands for plotting the ACF's of the bearing
# i need to add the body bearing autocorrelation (psi)
# and the reversal state (dpsi) autocorrelation

# plot postural covariance
traj.plotPosturalCovariance()

# plot postural mode distribution
traj.plotPosturalModeDistribution()

# plot postural time series - first posture
traj.plotPosturalTimeSeries(0)

# plot postural state space -- first and second posture
traj.plotPosturalPhaseSpace(0,1)

# in 3D
traj.plotPosturalPhaseSpace3D(0,1,2)

# as density
traj.plotPosturalPhaseSpaceDensity(0,1)

# ENSEMBLES
# make an ensemble of all the worms (6 in this case)
ens = wta.WormTrajectoryEnsemble([])
for worm in f['worms'][strain]:
	ens.append(wta.WormTrajectory(f, strain, worm))
ens = wta.WormTrajectoryEnsemble([traj for traj in ens if ~traj.allCentroidMissing and ~traj.allPostureMissing])
ens.sort()

# tile trajectory plots (can modify to do the same for any trajectory object plot)
ens.tilePlots(lambda t: t.plotTrajectory(showPlot=False))

# plot the mean squared displacement
ens.plotMeanSquaredDisplacement()

# plot the speed distribution
ens.plotSpeedDistribution()

# calculate postural measurements from the ensemble
ens.calculatePosturalMeasurements()

# plot the postural covariance
ens.plotPosturalCovariance()

# plot the postural mode distribution
ens.plotPosturalModeDistribution()

# plot the postural phase space density (need to update to use KDE)
ens.plotPosturalPhaseSpaceDensity(0,1)

# GROUPS
# divide the data into two ensembles
ens1 = ens[:3]
ens1.name = '0-2'
ens1.color = 'b'
ens2 = ens[3:]
ens2.name = '3-5'
ens2.color = 'r'
group = wta.WormTrajectoryEnsembleGroup([ens1, ens2])

# plot mean squared displacement
group.plotMeanSquaredDisplacement()

# plot speed distributions
group.plotSpeedDistribution()


# MODELS (only for single trajectories for now)
import wormtracker.models as wtm
m = wtm.Helms2014CentroidModel()  # Helms 2014 model

# fit reversal component
m.fitReversals(traj, plotFit=True)

# fit bearing drift
m.fitBearingDrift(traj, plotFit=True)

# fit bearing diffusion
m.fitBearingDiffusion(traj, plotFit=True)

# fit speed component
m.fitSpeed(traj, plotFit=True)
