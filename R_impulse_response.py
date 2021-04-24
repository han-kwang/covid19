#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:14:26 2021

@author: @hk_nien
"""


import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import pandas as pd
import tools
import nlcovidstats as nlcs


nlcs.init_data(autoupdate=True)


df1, _npop = nlcs.get_region_data('Nederland', lastday=-1, correct_anomalies=True)


df1['Impulse'] = 1
df1.iloc[-40, -1] = 1.01
df1['Impulse'] = scipy.signal.savgol_filter(df1['Impulse'].values, 7, 0)

np.random.seed(1)
df1['Noise'] = np.random.normal(1.0, scale=0.01, size=len(df1))
df1['Noise'] = scipy.signal.savgol_filter(df1['Noise'].values, 7, 0)


ndays = 200
# skip the first 10 days because of zeros
Rts = {} # key -> [Rt, Rt_sm, freqs, spec, spec_smooth]
for col in ['Delta7r', 'Impulse', 'Noise']:
    Rt, _ = nlcs.estimate_Rt_series(df1[col].iloc[10:], delay=7, Tc=4)
    Rt = Rt[-ndays:]
    Rsmooth = pd.Series(scipy.signal.savgol_filter(Rt.values, 13, 2, mode='interp'),
                      index=Rt.index)

    smooth_spec = np.fft.rfft(Rsmooth.values - 1)*100
    points_spec = np.fft.rfft(Rt.values - 1)*100
    freqs = np.arange(len(smooth_spec)) * (1/len(Rt))
    Rts[col] = [Rt, Rsmooth, freqs, points_spec, smooth_spec]

plt.close('all')

# Rts: key -> [Rt, Rt_sm, freqs, spec, spec_smooth, ax_title
Rts['Delta7r'] += [f'Rt Netherlands (t=0 at {Rt.index[-40].strftime("%Y-%m-%d")}']
Rts['Impulse'] += ['Rt response for 1 day with +1 % positive']
Rts['Noise'] += ['Rt calculated from random numbers (stdev=0.01)']


### Spectrum
fig, ax = plt.subplots(tight_layout=True, figsize=(7, 3))
freqs, points_spec, smooth_spec = Rts['Impulse'][2:5]

ax.plot(freqs, np.abs(points_spec)**2, 'o-', label='impulse response', markersize=2)
ax.plot(freqs, np.abs(smooth_spec)**2, 'o-', label='smooth impulse response', markersize=2)
ax.set_title('Frequency spectrum of smooth impulse response')
ax.set_xlabel('frequency (d$^{-1}$)')
ax.grid()
ax.legend()
fig.show()

#%%
ts = np.arange(ndays) - 160

fig, axs = plt.subplots(3, 1, tight_layout=True, figsize=(7, 7), sharex=True)
for ax, (Rt, Rsm, _, _, _, ax_title) in zip(axs, Rts.values()):
    ax.plot(ts, Rt.values, 'o', markersize=2, label='Daily points')
    ax.plot(ts, Rsm.values, label='Smoothed data')
    ax.grid()
    ax.legend(loc='upper right')
    ax.set_title(ax_title)
    ax.set_xlim(-35, 35)
    ax.set_xlabel('Day number')
fig.show()