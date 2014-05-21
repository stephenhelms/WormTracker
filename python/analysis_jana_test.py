import h5py
import os
os.chdir('D:\\Stephen\\Documents\\Code\\wormtracker-matlab\\python')
import wormtracker as wt
import analysis as wta
import matplotlib.pyplot as plt

# configuration
wta.configureMatplotLibStyle()
path = 'U:\\Koers\\worm_videos'

datasets = [(7, 'day_7\\2014-04-14_n2_a_b_day_7_processed.h5',
             'day_7'),
            (7, 'day_7\\2014-04-14_daf-2_a_b_day_7_processed.h5',
             'day_7'),
            (7, 'day_7\\2014-04-14_lin-44_a_b_day_7_processed.h5',
             'day_7')]
"""
            (8,
             'Recordings_experiment_1\\day_8\\merge_2014-04-15_n2_a_b_day_8_processed.h5',
             'Recordings_experiment_1\\day_8\\2014-04-15_n2_a_b_day_8.avi'),
            (8,
             'Recordings_experiment_1\\day_8\\merge_2014-04-15_daf-2_a_b_day_8_processed.h5',
             'Recordings_experiment_1\\day_8\\2014-04-15_daf-2_a_b_day_8.avi'),
            (8,
             'Recordings_experiment_1\\day_8\\merge_2014-04-15_lin-44_a_b_day_8_processed.h5',
             'Recordings_experiment_1\\day_8\\2014-04-15_lin-44_a_b_day_8.avi'),]
"""

# load all the data into a group
group = wta.WormTrajectoryEnsembleGroup([])
for dataset in datasets:
    f = h5py.File(os.path.join(path, dataset[1]), 'r')
    videoFile = os.path.join(path, dataset[2])
    strains = f['worms'].keys()
    for strain in strains:
        name = ' '.join([strain, 'Day', str(dataset[0])])
        ens = wta.WormTrajectoryEnsemble(
            [wta.WormTrajectory(f, strain, wormID,
                                videoFilePath=videoFile)
             for wormID in f['worms'][strain].keys()], name)
        ens.processAll()
        ens.sort()
        bad = [traj for traj in ens if (traj.allCentroidMissing or
                                        traj.allPostureMissing)]
        for traj in bad:
            ens.remove(traj)
        ens.calculatePosturalMeasurements()
        group.append(ens)
group.colorScheme = {
    group[0]: 'k',  # wild-type
    group[1]: 'b',  # daf-2
    group[2]: 'r',  # lin-44
}

# compare speed distributions
group.plotSpeedDistribution()  # update with grid, box off, square axis

# compare MSD
group.plotMeanSquaredDisplacement()  # same as above

# plot separately
plt.figure()
n = len(group)
ax0 = None
for i, ens in enumerate(group):
    if i == 0:
        ax0 = plt.subplot(1, n, i+1)
    else:
        plt.subplot(1, n, i+1, sharex=ax0, sharey=ax0)
    ens.plotSpeedDistribution(showPlot=False)
    plt.title(ens.name)
plt.show()
