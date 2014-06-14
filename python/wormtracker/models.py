import os
import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import itertools
import collections
from abc import ABCMeta, abstractmethod


class TrajectoryModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def toParameterVector(self):
        pass

    @abstractmethod
    def fromParameterVector(self, vector):
        pass

    @abstractmethod
    def fit(self, trajectory):
        pass

    def simulate(self, storeFile, location, nTimes=10):
        print 'Running simulations...'
        for i in xrange(nTimes):
            print '{0} of {1}'.format(i, nTimes)
            location = location + '/' + str(i)
            self._doSimulation(storeFile, location)

    @abstractmethod
    def _doSimulation(self, storeFile, h5loc):
        pass

    @abstractmethod
    def visualize(self, trajectory):
        pass

    def __str__(self):
        vector, labels, units = self.toParameterVector()
        return str({labels[i]: str(vector) + ' ' + units[i]
                    for i in xrange(len(vector))})


class Helms2014CentroidModel(TrajectoryModel):
    def __init__(self):
        # reversals
        self.tau_fwd = None
        self.tau_rev = None

        # bearing
        self.mu_omega = None
        self.D_psi = None

        # speed
        self.mu_s = None
        self.tau_s = None
        self.D_s = None
        self.sigma_s = None

    def fit(self, trajectory):
        self.fitReversals(trajectory)
        self.fitBearing(trajectory)
        self.fitSpeed(trajectory)

    def fitReversals(self, trajectory):
        raise NotImplemented()

    def fitBearing(self, trajectory):
        raise NotImplemented()

    def fitSpeed(self, trajectory):
        raise NotImplemented()

    def toParameterVector(self):
        return (np.array([self.tau_fwd,
                         self.tau_rev,
                         self.mu_omega,
                         self.D_psi,
                         self.mu_s,
                         self.tau_s,
                         self.D_s,
                         self.sigma_s]),
                [r'\tau_{fwd}', r'\tau_{rev}', r'\mu_\omega',
                 r'D_\psi', r'\mu_s', r'\tau_s', r'D_s', r'\sigma_s'],
                ['s', 's', 'rad/s', r'rad^2/s', r'\micro m/s',
                 's', r'(\micro m/s)^2 s^{-1}', r'\micro m/s'])

    def fromParameterVector(self, vector):
        # reversals
        self.tau_fwd = vector[0]
        self.tau_rev = vector[1]

        # bearing
        self.mu_omega = vector[2]
        self.D_psi = vector[3]

        # speed
        self.mu_s = vector[4]
        self.tau_s = vector[5]
        self.D_s = vector[6]
        self.sigma_s = vector[7]

    def _doSimulation(self, storeFile, location):
        raise NotImplemented()

    def visualize(self, trajectory):
        raise NotImplemented()
