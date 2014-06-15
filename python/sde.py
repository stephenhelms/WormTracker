import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import stats
import tsstats


class StochasticDifferentialEquation(object):
    def __init__(self, detFunc, stochFunc, params):
        self.detFunc = detFunc
        self.stochFunc = stochFunc
        self.params = params

    def evalDet(self, t, y):
        return self.detFunc(self.params, y, t)

    def evalStoch(self, t, y):
        return self.stochFunc(self.params, y, t)

    def integrateEuler(self, t0, tmax, dt, y0=0.):
        t = np.arange(t0, tmax, dt, float)
        y = np.zeros(t.shape)
        sqrtdt = np.sqrt(dt)
        r = np.random.standard_normal(t.shape)
        for i, ti in enumerate(t):
            if i==0:
                y[i] = y0
            else:
                detPart = self.evalDet(ti, y[i-1])
                stochPart = self.evalStoch(ti, y[i-1])
                y[i] = y[i-1] + detPart*dt + stochPart*sqrtdt*r[i-1]
        return t, y

    def plotSimulation(self, t0, tmax, dt, y0=0.,
                       plotStyle='k-', showPlot=True):
        t, y = self.integrateEuler(t0, tmax, dt, y0)
        plt.plot(t, y, plotStyle)
        plt.xlabel('Time (s)')
        plt.ylabel(r'$y(t)$')
        plt.grid(True)
        if showPlot:
            plt.show()

    def plotSimulationEnsemble(self, t0, tmax, dt, y0=0.,
                               nSims=10, lineColor=(0.5,0.5,0.5),
                               showPlot=True):
        for i in xrange(nSims):
            t, y = self.integrateEuler(t0, tmax, dt, y0)
            plt.plot(t, y, '-', color=lineColor)
            plt.hold(True)
        plt.xlabel('Time (s)')
        plt.ylabel(r'$y(t)$')
        plt.grid(True)
        if showPlot:
            plt.show()


class Fittable(object):
    def fit(t, y):
        raise NotImplemented()


class TheoreticalACF(object):
    def acf(self, tau):
        raise NotImplemented()

    def plotACFComparison(self, t, y, tau,
                          dataStyle='k.',
                          theoryStyle='r-',
                          showPlot=True):
        dts = t[1:] - t[:-1]
        dt = dts[0]
        if not np.all(dts - dt < 1e-12):
            raise Exception('Time must be evenly sampled.')
        lags = np.round(tau/dt)
        Cdata = tsstats.acf(y, lags)
        plt.plot(lags*dt, Cdata, dataStyle)
        Ctheory = self.acf(tau)
        plt.plot(tau, Ctheory, theoryStyle)
        plt.xlabel(r'$\tau$ (s)')
        plt.ylabel(r'$\langle y(t) \cdot y(t+\tau) \rangle / \langle y(t) \cdot y(t) \rangle$')
        plt.ylim((-1, 1))
        plt.xlim((tau.min(), tau.max()))
        plt.grid(True)
        if showPlot:
            plt.show()


class OrnsteinUhlenbeck(StochasticDifferentialEquation,
                        Fittable,
                        TheoreticalACF):
    def __init__(self, tau, mu, sigma):
        StochasticDifferentialEquation.__init__(self,
            lambda p, y, t: self._ouDetFunc(p['tau'],
                                            p['mu'],
                                            y),
            lambda p, y, t: self._ouStochFunc(p['sigma']),
            {'tau': tau, 'mu': mu, 'sigma': sigma})

    def _ouDetFunc(self, tau, mu, y):
        return 1./tau * (mu - y)

    def _ouStochFunc(self, sigma):
        return sigma

    def stationaryVariance(self):
        """
        Returns the theoretical stationary variance for an
        Ornstein-Uhlenbeck process:
        .. math:: \dfrac{1}{2} \sigma^2 \tau
        """
        return self.params['sigma']**2 / 2. * self.params['tau']

    def acf(self, tau):
        """
        Returns the theoretical autocorrelation function
        .. math:: \exp\left( -t/\tau \right)
        """
        return np.exp(-tau / self.params['tau'])

    def fit(t, y):
        dts = t[1:] - t[:-1]
        dt = dts[0]
        if not np.all(dts - dt < 1e-12):
            raise Exception('Time must be evenly sampled.')
        return OrnsteinUhlenbeck.fitAutoregressive(y, dt)

    def fitAutoregressive(y, dt):
        """
        Returns the Ornstein-Uhlenbeck parameters estimated from an
        autoregressive fit to the timeseries y sampled with rate dt
        ({param dict}, y prediction, y residuals)
        """
        mu = y.mean()
        yhat = y - mu
        slope, intercept, r, p, se = stats.linregress(yhat[:-1], yhat[1:])
        tau = 1./(1.-slope)*dt
        yhatpred = ma.array([yhat[:-1]*slope, ma.missing])
        yeps = yhat - yhatpred
        sigma = yeps.std() / np.sqrt(dt)
        return ({'tau': tau, 'mu': mu, 'sigma': sigma},
                yhatpred + mu,
                yeps)


class DiffusionDrift(StochasticDifferentialEquation,
                     Fittable):
    def __init__(self, k, D):
        StochasticDifferentialEquation.__init__(self,
            lambda p, y, t: self._ddDetFunc(p['k']),
            lambda p, y, t: self._ddStochFunc(p['D']),
            {'k': k, 'D': D})

    def _ddDetFunc(self, k):
        return k

    def _ddStochFunc(self, D):
        return D

    def fit(t, y):
        dts = t[1:] - t[:-1]
        dt = dts[0]
        if not np.all(dts - dt < 1e-12):
            raise Exception('Time must be evenly sampled.')
        # drift
        slope, intercept, r, p, se = stats.linregress(t, y)
        ypred = intercept + slope*t
        # diffusion
        yeps = y - ypred
        sigma = yeps.std()
        D = sigma**2 / (2. * dt)
        return ({'k': slope, 'D': D},
                ypred,
                yeps)
