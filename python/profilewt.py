import os
os.chdir('D:\\Stephen\\Documents\\Code\\wormtracker-matlab\\python')
import wormtracker as wt
import wormtracker.config as wtc
import wormtracker.parallel as wtp
import cProfile
import pstats
import glob

testDir = 'D:\\wormTest'
outDir = 'out'
configFile = 'short_test.yml'

with open(os.path.join(testDir, configFile), 'r') as f:
    wvs = wtc.loadWormVideos(f)

# take only the first
wv = wvs[0]


def test():
    wv.regions[0].process()


if __name__ == "__main__":
    for f in glob.glob(os.path.join(os.path.join(testDir, outDir),
                                    '*.*')):
        os.remove(f)
    cProfile.run('test()', 'region')
    with open(os.path.join(testDir, 'profiling.txt'), 'w+') as f:
        stats = pstats.Stats('region', stream=f)
        stats.strip_dirs()
        output = stats.sort_stats('cumulative').print_stats()
