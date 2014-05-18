import h5py
import wormtracker as wt
import analysis

# configuration
videoPath = 'U:\\Koers\\worm_videos\\day_7'
results = 'D:\\2014-04-14_n2_a_b_day_7_processed.h5'
f = h5py.File(results, 'r')

# make trajectory ensemble
ens = analysis.WormTrajectoryEnsemble([analysis.WormTrajectory(f, 'N2', str(i+1),
                                       videoFilePath=videoPath) for i in xrange(16)],
                                      'N2 Day 7')
ens.processAll()  # grabs data from the store file
ens.sort()  # sort by name

# show all trajectories
ens.tilePlots(lambda t: t.plotTrajectory(showPlot=False))
# show all speed time series
ens.tilePlots(lambda t: t.plotSpeed(showPlot=False))
# show ensemble speed distribution
ens.plotSpeedDistribution()
# show ensemble MSD
ens.plotMeanSquaredDisplacement()

# make a group
ens1 = ens[:8]
ens1.name = 'first'
ens2 = ens[8:]
ens2.name = 'second'
ensg = analysis.WormTrajectoryEnsembleGroup([ens1, ens2],
                                            name='Test Comparison',
                                            colorScheme={ens1: 'b',
                                                         ens2: 'r'})

# compare speed distributions
ensg.plotSpeedDistribution()
# compare speed ACFs
ensg.plotSpeedAutocorrelation()
# compare MSDs
ensg.plotMeanSquaredDisplacement()

f.close()