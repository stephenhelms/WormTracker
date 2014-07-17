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

# load results
import wormtracker.analysis as wta
wta.configureMatplotLibStyle()  # make prettier plots
f = h5py.File(storeFile, 'r')
strain = 'N2'
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
traj.plotSpeedDistribution(bins=np.linspace(0,500,100))

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
