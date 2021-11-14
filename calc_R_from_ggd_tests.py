#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimating R (Netherlands) based on GGD tests only. Experimental.

The dataset https://data.rivm.nl/covid-19/COVID-19_uitgevoerde_testen.csv
can be used to estimate R. It suffers much less from noise due
to administrative delays than the RIVM 'by municipality' or 'casus' datasets.
Weekday patterns are much more predictable.

Two methods, based on N[day_number] of positive tests, delay D, generation
interval Tg:

- "Week-on-week": R[i-D] = (N[i]/N[i-7]) ** (Tg/7)
- "Derivative of rolling-7":

   - N7[i] = N[i-3:i+4].mean()
   - R[i-D] = (N7[i+1] / N7[i-1]) ** (Tg/2)
     (implemented using derivative of log(N7).)

   Here, the last 3 entries of N7 can be estimated using historical
   N7[i]/N[i] ratios (default: previous 2 weeks). Because of the predictable
   weekday patterns in this dataset, this seems to work better than in the
   municipality dataset. Still not super accurate though; resulting R value
   typically +/- 0.1 off, occasionally much more.

Created on Sat Nov 13 22:26:05 2021

Author: @hk_nien on Twitter.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nlcovidstats as nlcs
import tools

# Clone of the github.com/mzelst/covid-19 repo.
test_path = '../mzelst-covid19-nobackup/data-rivm/tests'


def load_ggd_pos_tests(lastday=-1, pattern_weeks=2, Tgen=4.0):
    """Get latest GGD test data as DataFrame.

    Parameters:

    - lastday: data up to which date to use (-1 = most recent, -2=second most
      recent).
    - pattern_weeks: how many previous weeks to estimate weekday patterns.
    - Tgen: generation interval (days)

    Return DataFrame:

    - Index: Date_tested (timestamps at 12:00)
    - n_tested: number tested on that day (over entire country)
    - n_pos: number positive on that day
    - n_pos_7: 7-day average (last 3 days are estimated).
    - gfac_7: growth factor week-on-week for n_pos
    - R_wow: R based on week-on-week (no time offset)
    - R_d7r: R based on derivative of rolling-7 derivative
      (no time offset)

    Days with known bad data will have NaN.
    """
    fnames = sorted(Path(test_path).glob('rivm_daily_*.csv.gz'))
    fname = fnames[lastday] # latest
    df = pd.read_csv(fname).drop(columns='Version')
    print(f'Loaded {fname}')
    df.rename(columns={'Date_of_statistics': 'Date_tested',
                       'Tested_with_result': 'n_tested',
                       'Tested_positive': 'n_pos'},
              inplace=True)
    df['Date_tested'] = pd.to_datetime(df['Date_tested'])
    df.drop(columns='Date_of_report', inplace=True)
    df = df.groupby('Date_tested').sum()

    bad_dates = ['2021-02-07']
    df.loc[pd.to_datetime(bad_dates), ['n_pos', 'n_tested']] = np.nan

    # Handle 7-day effects to estimate last 3 days.
    df['n_pos_7'] = df['n_pos'].rolling(7, center=True).mean()
    nrows = len(df)
    for i0 in range(nrows-3, nrows):
        ii = np.arange(i0-7*pattern_weeks, i0, 7)
        idxs = df.index[ii]
        ratio = (df.loc[idxs, 'n_pos_7']/df.loc[idxs, 'n_pos']).mean()
        df.loc[df.index[i0], 'n_pos_7'] = df.loc[df.index[i0], 'n_pos']*ratio

    # Shift dates to mid-day
    df.index = df.index + pd.Timedelta(12, 'h')

    # R estimated based on week-on-week growth
    df['gfac_7'] = df['n_pos'] / df['n_pos'].shift(7)
    df['R_wow'] = df['gfac_7'] ** (Tgen/7)

    # R estimated on rolling average
    log_n = np.log(df['n_pos_7'])
    d_log_n = 0.5*(log_n.shift(-1) - log_n.shift(1))
    df['R_d7r'] = np.exp(d_log_n * Tgen)

    return df


def add_dataset(ax, Rs, delay_days, label, marker, color, err_fan=(7, 0.15)):
    """Add an R dataset with smoothed line.

    - err_fan (n, dR): specify error fan-out to last n-1 days up to +/- dR.
    """
    delay = pd.Timedelta(delay_days, 'd')
    ax.plot(
        Rs.index - delay, Rs,
        marker=marker, linestyle='none', color=color, label=label
        )


    # create smooth line.
    Rs.interpolate(inplace=True)  # for missing values in the middle

    # remove leading/trailing NaN
    Rs = Rs[Rs.notna()]

    from scipy.signal import savgol_filter
    Rsmooth = savgol_filter(Rs.values, 11, 2)
    ax.plot(Rs.index - delay, Rsmooth, '-', color=color)
    if err_fan:
        n, dR = err_fan
        Rsmooth1 = Rsmooth[-n:]
        fan = np.linspace(0, dR, n)
        ax.fill_between(Rs.index[-n:] - delay, Rsmooth1-fan, Rsmooth1+fan,
                        color=color, alpha=0.2)

if __name__ == '__main__':
    nlcs.init_data()
    dfR_rivm = nlcs.DFS['Rt_rivm'].copy()

    plt.close('all')


    #fig, ax = plt.subplots(figsize=(12, 4), tight_layout=True)

    delay_Rwow = pd.Timedelta(6.0, 'd')
    delay_Rd7r = pd.Timedelta(4, 'd')

    num_days = 100
    lastday = -1  # -1 for most recent
    # lastday = -10  # for testing.
    df = load_ggd_pos_tests(lastday=lastday, pattern_weeks=2)
    nlcs.plot_Rt(num_days, lastday=lastday, delay=nlcs.DELAY_INF2REP, ylim=(0.6, 1.5))
    fig = plt.gcf()
    ax = fig.get_axes()[0]

    ax.plot(dfR_rivm['R'], color='k')
    add_dataset(ax, df['R_wow'], 6.5, 'GGD week-op-week', 'x', '#008800', err_fan=None)
    add_dataset(ax, df['R_d7r'], 3.0, 'GGD afgeleide', '+', 'red', err_fan=None)

    ax.legend()

