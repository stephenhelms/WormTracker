import h5py
import os
os.chdir('D:\\Stephen\\Documents\\Code\\wormtracker-matlab\\python')
import wormtracker.analysis as wta
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

# load all trajectories
ensAll = wta.WormTrajectoryEnsemble([])
for dataset in datasets:
    f = h5py.File(os.path.join(path, dataset[1]), 'r')
    videoFile = os.path.join(path, dataset[2])
    strains = f['worms'].keys()
    for strain in strains:
        name = ' '.join([strain, 'Day', str(dataset[0])])
        trajs = [wta.WormTrajectory(f, strain, wormID,
                                    videoFilePath=videoFile)
                 for wormID in f['worms'][strain].keys()]
        for traj in trajs:
            traj.attr['day'] = dataset[0]
        bad = [traj for traj in trajs if (traj.allCentroidMissing or
                                          traj.allPostureMissing)]
        for traj in bad:
            trajs.remove(traj)
        ensAll.extend(trajs)

# all the strain and days
strains = set([traj.strain for traj in ensAll])
days = set([traj.attr['day'] for traj in ensAll])

# split ensemble by strain
enss = {strain: wta.WormTrajectoryEnsemble([traj for traj in ensAll
                                            if traj.strain == strain])
        for strain in strains}

# split ensemble by days
ensd = {day: wta.WormTrajectoryEnsemble([traj for traj in ensAll
                                        if traj.attr['day'] == day])
        for day in days}

# group by strain, ensemble by day
groups = {strain: wta.WormTrajectoryEnsembleGroup(
            [wta.WormTrajectoryEnsemble([traj for traj in ensAll
                                         if (traj.strain == strain and
                                             traj.attr['day'] == day)],
                                         strain + ' ' + str(day))
             for day in days])
          for strain in strains}

# group by day, ensemble by strain
groupd = {day: wta.WormTrajectoryEnsembleGroup(
            [wta.WormTrajectoryEnsemble([traj for traj in ensAll
                                         if (traj.strain == strain and
                                             traj.attr['day'] == day)],
                                         strain + ' ' + str(day))
             for strain in strains])
          for day in days}

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
