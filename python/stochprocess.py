import numpy as np
import numpy.ma as ma
from numpy import random
import matplotlib.pyplot as plt
from scipy import stats


class PoissonProcess(object):
    def __init__(self, tau):
        self.tau = tau

    def nextEvent(self):
        return random.exponential(self.tau, 1)[0]

    def eventsByTime(self, maxT):
        events = []
        ti = 0.
        while ti < maxT:
            ti += self.nextEvent()
            if ti <= maxT:
                events.append(ti)
        return np.array(events)

    def eventTimeSeries(self, maxT, dt):
        te = self.eventsByTime(maxT)
        ie = np.round(te / dt).astype(int)
        ts = np.zeros((np.round(maxT / dt),))
        ts[ie] = 1
        return ts

    @staticmethod
    def fit(tEvents):
        dtEvents = np.diff(tEvents)
        tau = stats.expon.fit(dtEvents)[1]
        return tau
