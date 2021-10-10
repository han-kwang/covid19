#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple simulation on fitting exponential growth on noisy discrete samples.

Background: growth of the AY.33 SARS-CoV-2 variant relative to other
lineages, in the Netherlands. So far, we have collected about 400 samples
of this lineage. Hypothesis is a logistic growth rate of 0.0336 per day.

Since AY.33 is only a small part of the total, we can approximate it as
exponential growth.

If n is the number of AY.33 cases on a date, then sqrt(n) is the standard
deviation (Poisson statistics). I will approximate the Poisson distribution
as a Gaussian distribution so that the rule of thumb ±2σ = 95% CI can be used.

2021-10-10 // Han-Kwang Nienhuys
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# How often are samples taken (in days). Exact value doesn't matter much.
dt_sample = 2

k = 0.0336*dt_sample

# the model is: n(t) = a0 * exp(k * t)
ts = np.arange(105/dt_sample)  # 105 days = 15 weeks
txs = np.arange(150/dt_sample)  # extended time series for extrapolation
ns = np.exp(k*ts)  # number of cases found at each time point.
Ntot = 400  # Total number of cases to be found.
a0 = Ntot/ns.sum()
ns *= a0
sigmas = np.sqrt(ns)

# Fitting exponentials is tricky, so take the log and do linear regression.
lgns = np.log(ns)
elgns = sigmas/ns
def linfunc(x, a0, a1):
    return a0+a1*x

# fitted log(a0), k values.
(fla0, fk), cov = curve_fit(linfunc, ts, lgns, absolute_sigma=True )

# a0, error a0
fa0 = np.exp(fla0)  # fit value for a0
ela0 = np.sqrt(cov[0, 0])   # error in log(a0)
ea0 = fa0 * ela0  # error in fit a0
print(f'Fit a0={fa0:.3g} ± {ea0:.3g}  (model: {a0:.3g})')

ek = np.sqrt(cov[1, 1])
print(f'Fit k={fk:.3g} ± {ek:.3g}  (model: {k:.3g})')


#%% generate curvves for this covariance matrix (monte-carlo)

# We want the 95% CI, so doing 20 Monte-Carlo runs should cover roughly
# the 95% CI.
n_mc = 20
# random distribution of log(a0) and k, each shape (n_mc)
np.random.seed(2)
la0mcs, kmcs = np.random.multivariate_normal([fla0, fk], cov, size=n_mc).T
curves = np.exp(la0mcs + kmcs*txs.reshape(-1, 1))  # shape (num_t, num_mc)

# sample realizations
# Apply the error to log(n), not n, so that we don't end up with negative
# n values.
ns_sampled = np.exp(lgns + np.random.normal(0, elgns, size=lgns.shape))


plt.close('all')
fig, ax = plt.subplots(tight_layout=True, figsize=(7, 4))
ax.errorbar(ts, ns_sampled, 2*sigmas, fmt='o', label='samples ± 2σ')
ax.plot(txs, curves[:, 0], color='gray', zorder=-10, alpha=0.2, linewidth=2,
        label='Realizations')
# other curves without label
ax.plot(txs, curves[:, 1:], color='gray', zorder=-10, alpha=0.2, linewidth=2)
ax.set_ylim(-1, 100)
ax.set_xlabel('Day number')
ax.set_ylabel('Number of cases')
ax.legend()

# ax.set_yscale('log')
fig.show()
