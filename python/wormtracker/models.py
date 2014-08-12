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
import scipy.optimize as opt
from tsstats import *


class TrajectoryModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def toParameterVector(self):
        pass

    @abstractmethod
    def fromParameterVector(self, vector):
        pass

    @abstractmethod
    def fit(self, trajectory, windowSize=None):
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
        self.k_psi = None
        self.D_psi = None

        # speed
        self.mu_s = None
        self.tau_s = None
        self.D_s = None

    def fit(self, trajectory, windowSize=None, plotFit=False):
        self.fitReversals(trajectory, windowSize, plotFit)
        self.fitBearing(trajectory, windowSize, plotFit)
        self.fitSpeed(trajectory, windowSize, plotFit)

    def fitReversals(self, trajectory, windowSize=None, plotFit=False):
        lags = np.arange(0, np.round(10.*trajectory.frameRate))
        if windowSize is None:
            dpsi = trajectory.getMaskedPosture(trajectory.dpsi)
            vdpsi = ma.array([np.cos(dpsi), np.sin(dpsi)]).T
            C = dotacf(vdpsi, lags)
        else:
            def getVectorDpsi(traj):
                dpsi = traj.getMaskedPosture(traj.dpsi)
                vdpsi = ma.array([np.cos(dpsi), np.sin(dpsi)]).T
                return vdpsi

            C = np.array([dotacf(getVectorDpsi(traj), lags)
                          for traj in trajectory.asWindows(windowSize)]).T
            C = C.mean(axis=1)
        tau = lags / trajectory.frameRate
        p, pcov = opt.curve_fit(self._reversalFitFunction, tau, C, [0, 0.5])
        f_rev = 0.5 - np.sqrt(p[1]/4)
        self.tau_rev = 10**p[0]/(1.-f_rev)
        self.tau_fwd = 10**p[0]/f_rev
        if plotFit:
            plt.plot(tau, C, 'k.')
            plt.plot(tau, self._reversalFitFunction(tau, p[0], p[1]), 'r-')
            plt.xlabel(r'$\tau$ (s)')
            plt.ylabel(r'$\langle \vec{\Delta\psi}(0) \cdot \vec{\Delta\psi}(\tau) \rangle$')
            textstr = '$\\tau_{\mathrm{rev}}=%.2f$ s\n$\\tau_{\mathrm{fwd}}=%.2f$ s'%(self.tau_rev, self.tau_fwd)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in lower left in axes coords
            ax = plt.gca()
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    horizontalalignment='right', verticalalignment='top', bbox=props)
            plt.show()

    def _reversalFitFunction(self, tau, log_tau_eff, Cinf):
        return (1.-Cinf)*np.exp(-tau/10**log_tau_eff) + Cinf

    def f_rev(self):
        return self.tau_rev / (self.tau_fwd + self.tau_rev)

    def model_dpsi_correlation(self, tau):
        log_tau_eff = np.log10(1. / (1./self.tau_fwd + 1./self.tau_rev))
        Cinf = 4.*(0.5 - self.f_rev())**2
        return self._reversalFitFunction(tau, log_tau_eff, Cinf)

    def fitBearing(self, trajectory, windowSize, plotFit=False):
        self.fitBearingDrift(trajectory, windowSize, plotFit)
        self.fitBearingDiffusion(trajectory, windowSize, plotFit)

    def fitBearingDrift(self, trajectory, windowSize=None, plotFit=False):
        lags = np.linspace(0, np.round(50.*trajectory.frameRate), 200)
        if windowSize is None:
            psi = unwrapma(trajectory.getMaskedPosture(trajectory.psi))
            D = drift(psi, lags)
        else:
            def result(traj):
                psi = unwrapma(trajectory.getMaskedPosture(trajectory.psi))
                return drift(psi, lags)

            D = np.array([result(traj)
                          for traj in trajectory.asWindows(windowSize)]).T
            D = D.mean(axis=1)
        tau = lags / trajectory.frameRate
        p = np.polyfit(tau, D, 1)
        self.k_psi = p[0]
        if plotFit:
            plt.plot(tau, D, 'k.')
            plt.plot(tau, np.polyval(p, tau), 'r-')
            plt.xlabel(r'$\tau$ (s)')
            plt.ylabel(r'$\langle \psi(\tau) - \psi(0) \rangle$ (rad)')
            textstr = '$k_\psi=%.2f$ rad/s'%(self.k_psi)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in lower left in axes coords
            ax = plt.gca()
            ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=14,
                    horizontalalignment='right', verticalalignment='bottom', bbox=props)
            plt.show()

    def fitBearingDiffusion(self, trajectory, windowSize=None, plotFit=False):
        tau, C = trajectory.getBodyBearingAutocorrelation(maxT=50.,
                                                          windowSize=windowSize)
        p, pcov = opt.curve_fit(self._bearingDiffusionFitFunction, tau, C, [-1.])
        self.D_psi = 10**p[0]
        if plotFit:
            plt.plot(tau, C, 'k.')
            plt.plot(tau, self._bearingDiffusionFitFunction(tau, p[0]), 'r-')
            plt.xlabel(r'$\tau$ (s)')
            plt.ylabel(r'$\langle \vec{\psi}(0) \cdot \vec{\psi}(\tau) \rangle$')
            textstr = '$D_\psi=%.2f \mathrm{rad}^2/\mathrm{s}$'%(self.D_psi)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in lower left in axes coords
            ax = plt.gca()
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    horizontalalignment='right', verticalalignment='top', bbox=props)
            plt.show()

    def _bearingDiffusionFitFunction(self, tau, log_D):
        return np.exp(-(10**log_D)*tau)

    def model_bearing_correlation(self, tau):
        return self._bearingDiffusionFitFunction(tau, np.log10(self.D_psi))

    def fitSpeed(self, trajectory, windowSize=None, plotFit=False):
        tau, C = trajectory.getSpeedAutocorrelation(maxT=10., windowSize=windowSize)
        p, pcov = opt.curve_fit(self._speedFitFunction, tau, C, [0., 3.])
        self.tau_s = 10**p[0]
        self.D_s = 10**p[1]
        s = trajectory.getMaskedCentroid(trajectory.s)
        s[trajectory.nearRev] = ma.masked
        self.mu_s = s.mean()
        if plotFit:
            plt.plot(tau, C, 'k.')
            plt.plot(tau, self._speedFitFunction(tau, p[0], p[1]), 'r-')
            plt.xlabel(r'$\tau$ (s)')
            plt.ylabel(r'$\langle \hat{s}(0) \cdot \hat{s}(\tau) \rangle \mathrm{(um/s)}^2$')
            textstr = '$\\tau_s=%.2f \mathrm{s}$\n$D_s=%.2f \mathrm{(um/s)}^2/\mathrm{s}$'%(self.tau_s, self.D_s)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in lower left in axes coords
            ax = plt.gca()
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    horizontalalignment='right', verticalalignment='top', bbox=props)
            plt.show()

    def _speedFitFunction(self, tau, log_tau, log_D):
        return (10**log_D)*(10**log_tau)*np.exp(-tau/10**log_tau)

    def model_s_correlation(self, tau):
        return _speedFitFunction(tau, np.log10(self.tau_s), np.log10(self.D_s))

    def toParameterVector(self):
        return (np.array([self.tau_fwd,
                         self.tau_rev,
                         self.k_psi,
                         self.D_psi,
                         self.mu_s,
                         self.tau_s,
                         self.D_s]),
                [r'\tau_{fwd}', r'\tau_{rev}', r'k_\psi',
                 r'D_\psi', r'\mu_s', r'\tau_s', r'D_s'],
                ['s', 's', 'rad/s', r'rad^2/s', r'\micro m/s',
                 's', r'(\micro m/s)^2 s^{-1}'])

    def fromParameterVector(self, vector):
        # reversals
        self.tau_fwd = vector[0]
        self.tau_rev = vector[1]

        # bearing
        self.k_psi = vector[2]
        self.D_psi = vector[3]

        # speed
        self.mu_s = vector[4]
        self.tau_s = vector[5]
        self.D_s = vector[6]

    def _doSimulation(self, storeFile, location):
        raise NotImplemented()

    def visualize(self, trajectory):
        raise NotImplemented()
