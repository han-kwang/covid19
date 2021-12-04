#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 14:02:04 2021

@author: hk_nien
"""
import re
import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
from tools import set_xaxis_dateformat

def load_tvt_data():
    """Return DataFrame with index date (mid-week 12:00), num_test, num_pos, f_pos."""

    records = []
    with open('data/TvT.txt') as f:
        for li in f.readlines():
            if li.startswith('#') or len(li) < 2:
                continue
            # typical line: "22-03-2021 - 28-03-2021 5081 16 0.3"
            fields = li.split()
            dates = [
                pd.to_datetime(fields[i], format='%d-%m-%Y')
                for i in [0, 2]
                ]
            n_test = int(fields[3])
            n_pos = int(fields[4])
            date_mid = dates[0] + (dates[1]-dates[0])/2 + pd.Timedelta('12 h')
            records.append((date_mid, dates[0], dates[1], n_test, n_pos))

    df = pd.DataFrame.from_records(
        records, columns=['Date_mid', 'Date_a', 'Date_b', 'num_test', 'num_pos']
        )

    if df.iloc[-1]['Date_b'] < pd.to_datetime('now') - pd.to_timedelta('9 d, 15:15:00'):
        print(
            '** Warning: TvT data may be outdated. Update data/TvT.txt from '
            'RIVM weekly report at '
            'https://www.rivm.nl/coronavirus-covid-19/actueel/'
            'wekelijkse-update-epidemiologische-situatie-covid-19-in-nederland .'
            )

    df = df.set_index('Date_mid')
    df['f_pos'] = df['num_pos'] / df['num_test']

    return df


def get_R_from_TvT():
    """Return DataFrame with R estimate from TvT data.

    Return DataFrame:

    - index: datetime index (12:00)
    - R: R estimate (one per week)
    - R_err: estimated R error (2sigma), one per week.
    - R_interp: interpolated R values (daily)
    """
    df = load_tvt_data()
    date0 = df.index[0]

    # ts: day number since start date
    ts = (df.index - date0) / pd.Timedelta('1 d')
    fposs = df['f_pos'].to_numpy()
    # convert week-over-week growth to R
    Tgen = 4.0 # generation interval
    Rs = (fposs[1:] / fposs[:-1]) ** (Tgen/(ts[1:] - ts[:-1]))

    # error estimate
    fposs_rel_err = 1 / np.sqrt(df['num_pos'].to_numpy())
    Rs_err = 2 * np.sqrt(fposs_rel_err[1:]**2 + fposs_rel_err[:-1]**2) * (Tgen/7)

    delay = 7.0 # delay from f_pos growth to R
    dates_R = df.index[1:] - pd.Timedelta(delay, 'd')

    Rs = pd.Series(Rs, index=dates_R)

    # Interpolated

    dates_i = pd.date_range(Rs.index[0], Rs.index[-1], freq='1 d')
    Rsi = pd.Series(np.nan, index=dates_i)
    Rsi.loc[dates_R] = Rs
    Rsi.interpolate('quadratic', inplace=True)


    dfR = pd.DataFrame(index=Rsi.index)
    dfR.loc[dates_R, 'R'] = Rs
    dfR.loc[dates_R, 'R_err'] = Rs_err * 2
    dfR['R_interp'] = Rsi
    return dfR

if __name__ == '__main__':
    df = get_R_from_TvT()

    plt.close('all')
    fig, ax = plt.subplots()
    ax.plot(df['R_interp'])
    ax.scatter(df.index, df['R'])
    set_xaxis_dateformat(ax)
    fig.show()

