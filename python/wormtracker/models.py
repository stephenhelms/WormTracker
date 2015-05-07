import os
import numpy as np
import numpy.ma as ma
from numpy import linalg as LA
import matplotlib.pyplot as plt
import h5py
import itertools
import collections
from abc import ABCMeta, abstractmethod
import scipy.optimize as opt
import scipy.integrate as scint
import scipy.sparse.linalg as SLA
import statsmodels.api as sm
import lmfit
from tsstats import *
import sde
import stochprocess


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
        self._prepareSimulation(storeFile, location, nTimes)
        print 'Running simulations...'
        for i in xrange(nTimes):
            print '{0} of {1}'.format(i, nTimes)
            self._doSimulation(storeFile, location, i)

    @abstractmethod
    def _prepareSimulation(self, storeFile, h5loc, nTimes):
        pass

    @abstractmethod
    def _doSimulation(self, storeFile, h5loc, i):
        pass

    @abstractmethod
    def visualize(self, trajectory, storeFile, location):
        pass

    def __str__(self):
        vector, labels, units = self.toParameterVector()
        return str({labels[i]: str(vector) + ' ' + units[i]
                    for i in xrange(len(vector))})


class Helms2014CentroidModel(TrajectoryModel):
    Helms2014_mean_traits = np.array([1.9327, 0.0642, 2.7815, -1.4129, -1.3353, 1.0023, 0.3089])
    Helms2014_mode1 = np.array([0.2729, 0.2871, -0.0554, 0.1470, 0.1754, 0.7188, 0.5206])

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

        self.length = 1

        # simulation settings
        self.duration = 30.*60.
        self.dt = 1./11.5

    def fit(self, trajectory, windowSize=None, plotFit=False):
        self.fitReversals(trajectory, windowSize, plotFit)
        self.fitBearing(trajectory, windowSize, plotFit)
        self.fitSpeed(trajectory, windowSize, plotFit)
        self.length = trajectory.length

    def fitReversals(self, trajectory, windowSize=None, plotFit=False):
        lags = np.arange(0, np.round(10.*trajectory.frameRate))
        if windowSize is None:
            dpsi = trajectory.getMaskedPosture(trajectory.dpsi)
            vdpsi = ma.array([ma.cos(dpsi), ma.sin(dpsi)]).T
            C = dotacf(vdpsi, lags, trajectory.excluded)
        else:
            def getVectorDpsi(traj):
                dpsi = traj.getMaskedPosture(traj.dpsi)
                if float(len(dpsi.compressed()))/float(len(dpsi)) > 0.2:
                    vdpsi = ma.array([ma.cos(dpsi), ma.sin(dpsi)]).T
                    return vdpsi
                else:
                    return ma.zeros((len(dpsi), 2))*ma.masked

            C = ma.array([dotacf(getVectorDpsi(traj), lags, traj.excluded)
                          for traj in trajectory.asWindows(windowSize)]).T
            C = C.mean(axis=1)
        tau = lags / trajectory.frameRate

        # do the bounded fit
        params = lmfit.Parameters()
        params.add('log_tau_eff', value=0.)
        params.add('Cinf', value=0.5, min=0., max=1.)

        if C.compressed().shape[0]>0:
            p = lmfit.minimize(self._reversalFitResidual, params, args=(tau, C))

            f_rev = 0.5 - np.sqrt(params['Cinf']/4)
            self.tau_rev = 10**params['log_tau_eff']/(1.-f_rev)
            self.tau_fwd = 10**params['log_tau_eff']/f_rev
        else:
            self.tau_rev = ma.masked
            self.tau_fwd = ma.masked
        if plotFit:
            plt.plot(tau, C, 'k.')
            plt.plot(tau, self._reversalFitFunction(tau, params['log_tau_eff'], params['Cinf']), 'r-')
            plt.xlabel(r'$\tau$ (s)')
            plt.ylabel(r'$\langle \vec{\Delta\psi}(0) \cdot \vec{\Delta\psi}(\tau) \rangle$')
            textstr = '$\\tau_{\mathrm{rev}}=%.2f$ s\n$\\tau_{\mathrm{fwd}}=%.2f$ s'%(self.tau_rev, self.tau_fwd)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in lower left in axes coords
            ax = plt.gca()
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    horizontalalignment='right', verticalalignment='top', bbox=props)
            plt.show()

    def _reversalFitResidual(self, vars, tau, data):
        log_tau_eff = float(vars['log_tau_eff'])
        Cinf = float(vars['Cinf'])
        model = self._reversalFitFunction(tau, log_tau_eff, Cinf)
        return (data-model)

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
        tau = lags / trajectory.frameRate
        if windowSize is None:
            psi = unwrapma(trajectory.getMaskedPosture(trajectory.psi))
            D = drift(psi, lags, trajectory.excluded)
            p = np.polyfit(tau, D, 1)
            self.k_psi = p[0]
        else:
            def result(traj):
                psi = traj.getMaskedPosture(traj.psi)
                if float(len(psi.compressed()))/float(len(psi)) > 0.2:
                    psi = unwrapma(psi)
                    return ma.array(drift(psi, lags, traj.excluded))
                else:
                    return ma.zeros((len(lags),))*ma.masked

            D = ma.array([result(traj)
                          for traj in trajectory.asWindows(windowSize)])
            k = np.array([np.polyfit(tau, Di, 1)[0]
                          for Di in D
                          if Di.compressed().shape[0]>50])
            self.k_psi = ma.abs(k).mean()
            D = ma.abs(D).T.mean(axis=1)
        
        
        if plotFit:
            plt.plot(tau, D, 'k.')
            plt.plot(tau, self.k_psi*tau, 'r-')
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
        return (np.array([self.mu_s,
                         self.tau_s,
                         self.D_s,
                         self.k_psi,
                         self.D_psi,
                         self.tau_fwd,
                         self.tau_rev,
                         self.length]),
                [r'\mu_s', r'\tau_s', r'D_s', 
                 r'k_\psi', r'D_\psi',
                 r'\tau_{fwd}', r'\tau_{rev}', r'length'],
                [r'\micro m/s', 's', r'(\micro m/s)^2 s^{-1}',
                 'rad/s', r'rad^2/s',
                 's', 's', 'um'])

    def fromParameterVector(self, vector):
        # speed
        self.mu_s = vector[0]
        self.tau_s = vector[1]
        self.D_s = vector[2]

        # bearing
        self.k_psi = vector[3]
        self.D_psi = vector[4]

        # reversals
        self.tau_fwd = vector[5]
        self.tau_rev = vector[6]

    def parametersToMode(self):
        p, labels, units = self.toParameterVector()
        norm = (np.log10(np.abs(p))-self.Helms2014_mean_traits)
        return np.dot(norm, self.Helms2014_mode1)

    def _prepareSimulation(self, storeFile, location, nTimes):
        with h5py.File(storeFile) as f:
            if location not in f:
                f.create_group(location)
            g = f[location]

            n = self.duration/self.dt
            if 's' not in g:
                g.create_dataset('s', (nTimes, n), dtype='float64')
            if 'psi' not in g:
                g.create_dataset('psi', (nTimes, n), dtype='float64')
            if 'phi' not in g:
                g.create_dataset('phi', (nTimes, n), dtype='float64')
            if 'dpsi' not in g:
                g.create_dataset('dpsi', (nTimes, n), dtype='float64')
            if 'v' not in g:
                g.create_dataset('v', (nTimes, n, 2), dtype='float64')
            if 'X' not in g:
                g.create_dataset('X', (nTimes, n, 2), dtype='float64')
            if 't' not in g:
                g.create_dataset('t', (n,), dtype='float64')

            t = np.arange(0, n)*self.dt
            g['t'][...] = t

    def _doSimulation(self, storeFile, location, i):
        with h5py.File(storeFile) as f:
            g = f[location]

            s = self._simulateSpeed()
            g['s'][i, :] = s
            psi = self._simulateBodyBearing()
            g['psi'][i, :] = psi
            r = self._simulateReversals()
            dpsi = np.zeros((r.shape[0],))
            dpsi[~r] = np.pi
            g['dpsi'][i, :] = dpsi
            phi = psi + dpsi
            g['phi'][i, :] = phi

            v = (s*np.array([np.cos(phi), np.sin(phi)])).T
            g['v'][i, ...] = v
            X = np.zeros((r.shape[0], 2))
            X[1:,:] = scint.cumtrapz(v, dx=self.dt, axis=0)
            g['X'][i, ...] = X

    def _simulateSpeed(self):
        ou = sde.OrnsteinUhlenbeck(self.tau_s, self.mu_s, np.sqrt(2.*self.D_s),
                                   positive=True)
        return ou.integrateEuler(0., self.duration, self.dt, self.mu_s)[1]

    def _simulateBodyBearing(self):
        dd = sde.DiffusionDrift(self.k_psi, self.D_psi)
        return dd.integrateEuler(0, self.duration, self.dt, 0.)[1]

    def _simulateReversals(self):
        psp = stochprocess.PoissonTwoStateProcess(self.tau_fwd, self.tau_rev)
        return psp.stateTimeSeries(self.duration, self.dt)

    def visualize(self, trajectory, storeFile, location):
        # plots: X, msd, vacf, Cdpsi, Cpsi, Cs, mean bearing vs time, speed dist
        # Xtraj msd  C mean bearing
        # Xsim  vacf   speed dist
        # trajectories
        self.visualizeTrajectory(trajectory, storeFile, location,
                                 axes=(plt.subplot(4, 2, 0),
                                       plt.subplot(4, 2, 4)),
                                 showPlot=False)
        # mean-squared displacement
        self.visualizeMeanSquaredDisplacement(trajectory, storeFile,
                                              location,
                                              plt.subplot(4, 2, 1),
                                              showPlot=False)
        
        plt.show()

    def visualizeTrajectory(self, trajectory, storeFile, location,
                            axes=None, showPlot=True):
        with h5py.File(storeFile, 'r') as f:
            g = f[location]
            if axes is None:
                axes = (plt.subplot(211), plt.subplot(212))
            dataAx, simAx = axes
            plt.sca(dataAx)
            trajectory.plotTrajectory(showPlot=False)
            plt.title('Observed Trajectory')
            simAx.plot(g['X'][0, :, 0], g['X'][0, :, 1], 'k-')
            simAx.xlabel('x (um)')
            simAx.ylabel('y (um)')
            simAx.gca().set_aspect('equal')
            simAx.title('Simulated Trajectory')
        if showPlot:
            plt.show()

    def visualizeMeanSquaredDisplacement(self, trajectory, storeFile,
                                         location, axes=None, showPlot=True):
        with h5py.File(storeFile, 'r') as f:
            g = f[location]
            if axes is not None:
                plt.sca(axes)
            tau = np.logspace(-1, 2, 200)
            tau, Sigma = trajectory.getMeanSquaredDisplacement(tau)
            plt.plot(np.log10(tau), Sigma, 'k.')
            lags = np.round(tau/self.dt)
            Sigma = ma.zeros((g['X'].shape[0], tau.shape[0]))
            for j, X in enumerate(g['X']):
                for i, lag in enumerate(lags):
                    displacements = X[lag:, :] - X[:-lag, :]
                    Sigma[j, i] = np.mean(np.log10(np.sum(displacements**2,
                                                          axis=1)))
            mu, cl, cu = bootstrap(Sigma)
            plt.plot(np.log10(tau), mu, '.-', color='r')
            plt.fill_between(log10(tau), cl, cu, facecolor='r', alpha=0.3)

            plt.xlabel(r'log $\tau$ \ (s)')
            plt.ylabel(r'log $\langle \| x(t) - x(t-\tau) \|^2 \rangle$ (um^2)')
            plt.xlim((np.min(np.log10(tau)), np.max(np.log10(tau))))
            plt.grid(True)
        if showPlot:
            plt.show()


class Stephens2014PostureDynamicsModel(TrajectoryModel):
    def __init__(self):
        self.vPostures = None
        self.nPostures = 4
        self.order = 0
        self.tau = None
        self.foscil = None
        self.vDynamics = None
        self.lDynamics = None
        self.damp = None
        self.robust = False

    def fit(self, trajectory, windowSize=None, plotFit=False):
        posture = trajectory.getMaskedPosture(trajectory.posture)
        dt = 1./trajectory.frameRate
        vPostures = self.vPostures
        if vPostures is None:
            vPostures = trajectory.vtheta[:, :self.nPostures]
        modes = np.dot(posture, vPostures)
        modes[posture.mask.any(axis=1), :] = ma.masked
        # normalize modes to zeros mean (almost true anyway and unit variance)
        modes = (modes - modes.mean(axis=0))/modes.std(axis=0)
        # window size not supported yet
        t = trajectory.t[self.order+1:]
        y = modes[self.order+1:, :]
        x = ma.array([modes[i:-1, :] for i in xrange(self.order+1)]).squeeze()
        if windowSize is None:
            w, nu, v, l, foscil, tau = self._fitWindow(x, y, dt)
            self.tw = None
        else:
            w = []
            nu = []
            v = []
            l = []
            foscil = []
            tau = []
            tw = []
            for i in xrange(windowSize/2, t.shape[0]-windowSize/2,
                            windowSize/10):
                xw = x[i-windowSize/2:i+windowSize/2]
                yw = y[i-windowSize/2:i+windowSize/2]
                if (xw.compressed().shape[0] > 0.75*windowSize and
                        yw.compressed().shape[0] > 0.75*windowSize):
                    tw.append(t[i])
                    f = self._fitWindow(xw, yw, dt)
                    w.append(f[0])
                    nu.append(f[1])
                    v.append(f[2])
                    l.append(f[3])
                    foscil.append(f[4])
                    tau.append(f[5])
            self.tw = tw
        self.w = w
        self.nu = nu
        self.v = v
        self.l = l
        self.foscil = foscil
        self.tau = tau

    def _fitWindow(self, x, y, dt):
        sel = ~y.mask.any(axis=1) & ~x.mask.any(axis=1)
        if self.damp is None:
            if ~self.robust:
                f = LA.lstsq(x[sel, :], y[sel, :])
            else:
                resrlm = sm.RLM(y[sel, :], x[sel, :]).fit()
                w = resrlm.params
        else:
            f = SLA.lsmr(y[sel, :], x[sel, :], self.damp)
        w = f[0]
        pred = np.dot(x, w)
        eps = y - pred
        nu = eps[sel, :].std(axis=0)/np.sqrt(dt)
        # timescales
        l, v = LA.eig(w)
        theta = np.arctan2(np.imag(l), np.real(l))
        foscil = (np.abs(theta)/dt)/(2.*np.pi)  # in Hz
        tau = -1.*np.sign(np.real(l))/(np.log(np.abs(l))/dt)  # in seconds
        return (w, nu, v, l, foscil, tau)

    def toParameterVector(self):
        pass

    def fromParameterVector(self, vector):
        pass

    def _prepareSimulation(self, storeFile, h5loc, nTimes):
        pass

    def _doSimulation(self, storeFile, h5loc, i):
        pass

    def visualize(self, trajectory, storeFile, location):
        pass

    def plotDynamicsSpace(self):
        plt.scatter(np.array(self.tau), np.array(self.foscil))
        plt.xlabel('Damping Time (s)')
        plt.ylabel('Oscillation Frequency (Hz)')
        plt.show()

    def plotDynamicsSpaceDensity(self, dampRange=(0, 30), oscilRange=(0, 0.5)):
        X, Y = np.mgrid[dampRange[0]:dampRange[1]:100j,
                        oscilRange[0]:oscilRange[1]:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([np.array(self.tau).flatten(), np.array(self.foscil).flatten()])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        plt.imshow(np.rot90(Z), extent=[dampRange[0],dampRange[1],oscilRange[0],oscilRange[1]],
                   aspect='auto')
        plt.xlabel('Damping Time (s)')
        plt.ylabel('Oscillation Frequency (Hz)')
        plt.colorbar()
        plt.show()

    def plotDynamicsTime(self):
        ax1 = plt.subplot(211)
        plt.plot(np.array(self.tw), np.array(self.tau), '.')
        plt.xlabel('Time (s)')
        plt.ylabel('Damping Time (s)')
        plt.subplot(212, sharex=ax1)
        plt.plot(np.array(self.tw), np.array(self.foscil), '.')
        plt.xlabel('Time (s)')
        plt.ylabel('Oscillation Frequency (Hz)')
        plt.show()

    def _doSimulation(self, storeFile, location):
        raise NotImplemented()
