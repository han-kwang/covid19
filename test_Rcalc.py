#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testing R calculation method on hypothetical scenarios.
Created on Mon Mar 22 08:38:39 2021

@author: @hk_nien
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

def calc_R_from_cases(daily_pos, delay=7, Tc=4.0):
    """Calculate from daily-cases array.

    Return a dataframe with:

    - npos: number of positives per day
    - npos7: same, 7-day average.
    - Rt: R estimate.
    """

    npos = pd.Series(daily_pos)
    npos7 = npos.rolling(7, center=True).mean()


    log_r = np.log(npos7.to_numpy()) # shape (n,)
    log_slope = (log_r[2:] - log_r[:-2])/2 # (n-2,)
    Rt = np.exp(Tc*log_slope) # (n-2,)

    index = npos.index[1:-1] - delay
    Rts = pd.Series(index=index, data=Rt, name='Rt').iloc[3:]

    df = pd.DataFrame(dict(npos=npos, npos7=npos7, Rt=Rts))

    return df

def add_noise(npos, f=0.05, seed=2):
    """Add noise, stdev = f*value, truncated distribution"""

    np.random.seed(seed)
    lam = 1/f**2
    noise_fac = np.random.poisson(lam, size=npos.shape)/lam

    # redo outliers
    mask = (np.abs(noise_fac-1) > 2*f)
    noise_fac[mask] = np.random.poisson(lam, size=np.count_nonzero(mask))/lam



    return npos * noise_fac


def calc_plot_scenario(daily_pos, labels=None, title='Simulatie'):
    """daily_pos: 1D array of daily case numbers.

    labels: optional list (day_number, label).

    Return dataframe.
    """

    df = calc_R_from_cases(daily_pos)
    fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True, figsize=(7, 6))
    ax = axs[0]

    ax.set_title(f'{title}\n\nDagelijkse positieve tests')

    psize = max(15, 1500/len(df))

    ax.scatter(df.index, df['npos'], label='Dagcijfers', s=psize)
    ax.plot(df['npos7'], label='7-d gemiddelde')
    ax.legend()
    ax.grid()


    ax = axs[1]
    ax.plot(df['Rt'])
    ax.grid()
    ax.set_title('Schatting reproductiegetal Rt')
    ax.set_xlabel('Dagnummer')


    for ax, column in zip(axs, ['npos', 'Rt']):

        ys = df[column]
        if ys.min() < 0.1*ys.max():
            ax.set_yscale('log')
            ax.grid(axis='y', which='minor')
            continue

        ymin, ymax = ax.get_ylim()
        if ymin>0 and ymax/ymin < 2:
            ymid = (ymax+ymin)/2
            ymin, ymax = ymid*0.5, ymid*1.5
            ax.set_ylim(ymin, ymax)


    if labels:
        ymins = [ax.get_ylim()[0] for ax in axs]
        for (x, txt) in labels:
            for ax, ymin in zip(axs, ymins):
                t = ax.text(
                    x, ymin+0.01, f' {txt}', rotation=90,
                    horizontalalignment='center',
                    verticalalignment='bottom'
                    )
                t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))


    ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
    fig.canvas.set_window_title(title)
    fig.show()
    plt.pause(0.3)
    return df

def scenario_constant():
    npos = add_noise(np.full(70, 7000))
    calc_plot_scenario(npos, title='Constant')

def scenario_growth():

    R, Tc = 1.1, 4.0
    npos = 1000 * R ** (np.arange(70)/Tc)
    npos = add_noise(npos)
    calc_plot_scenario(npos, title='Scenario: constant R=1.1')


def scenario_more_tests():
    npos = np.full(70, 7000.0)
    npos[21:] *= 1.50
    npos = add_noise(npos)
    calc_plot_scenario(
        npos,
        title='Scenario: stapsgewijze toename testbereidheid (+50%)',
        labels=[(21, '+50% testbereidheid')]
        )


def scenario_single_mass_test():
    npos = np.full(70, 7000.0)
    npos[21] = 150000
    npos = add_noise(npos)
    df = calc_plot_scenario(
        npos,
        title='Scenario: eenmalige massa-test',
        labels=[(21, 'testdag')]
        )
    i = df['Rt'].argmin()
    print(df.iloc[i-2:i+3])
    return df

def scenario_weekday_effects():

    R, Tc = 1.08, 4.0
    npos = 3910 * R ** (np.arange(49)/Tc)
    wday_mul = 1/np.array([1.12, 1.16, 0.99, 1.  , 0.92, 0.85, 0.96]) # 2 wk
    wday_mul = 1/np.array([1.04, 1.21, 1.03, 1.02, 0.9 , 0.87, 0.94]) # 1 wk
    #wday_mul = 1/np.array([1.14, 1.18, 0.99, 0.96, 0.89, 0.88, 0.97]) # 5 wk
    # wday_mul = 1/np.array([1.16, 1.17, 0.97, 0.93, 0.89, 0.89, 0.98]) # 7 wk
    npos_7 = npos.reshape(-1, 7)
    npos_7 *= wday_mul
    npos = npos[:-6]
    calc_plot_scenario(npos, title='Scenario: groei met weekdageffecten')



plt.close('all')

#scenario_constant()
#scenario_more_tests()
#scenario_growth()
scenario_weekday_effects()
#df = scenario_single_mass_test()








