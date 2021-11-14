#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 21:46:26 2021

@author: hk_nien @ Twitter
"""
from pathlib import Path
from time import time
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tools import set_xaxis_dateformat


# Clone of the github.com/mzelst/covid-19 repo.
test_path = '../mzelst-covid19-nobackup/data-rivm/tests'

_CACHE = {}  # store slow data here: (date_a, date_b) -> (tm, dataframe)
#%%
def load_testdata(min_date='2021-01-01', max_date='2099-01-01'):
    """Return DataFrame. Specify mindate as 'yyyy-mm-dd'.

    Use cache if available.
    Pull data from mzelst-covid19 repository (which should be up to date).

    Return DataFrame with multi-index (sdate, fdate).
    """
    if (min_date, max_date) in _CACHE:
        tm, df = _CACHE[(min_date, max_date)]
        if time() - tm < 3600:
            print('Loaded from cache.')
            return df.copy()

    fnames = sorted(Path(test_path).glob('rivm_daily_*.csv.gz'))
    dfs = []
    for fn in fnames:
        m = re.search(r'_(\d\d\d\d-\d\d-\d\d).csv.gz', str(fn))
        if not m:
            continue
        fdate = m.group(1)
        if fdate < min_date or fdate > max_date:
            continue
        df = pd.read_csv(fn).drop(columns='Version')
        fdate = df.iloc[0]['Date_of_report']
        df = df.groupby('Date_of_statistics').sum()
        df.reset_index(inplace=True)
        df['Date_of_report'] = fdate
        dfs.append(df)

    df = pd.concat(dfs)

    colmap = {
        'Date_of_report': 'fdate',
        'Date_of_statistics': 'sdate',
        'Security_region_code': 'srcode',
        'Tested_with_result': 'n_tested',
        'Tested_positive': 'n_pos'
        }

    # df.drop(columns=['Security_region_name', 'Version'], inplace=True)
    df.rename(columns=colmap, inplace=True)
    df['fdate'] = pd.to_datetime(df['fdate'].str.slice(0, 10))
    df['sdate'] = pd.to_datetime(df['sdate'])
    df = df[['sdate', 'fdate', 'n_tested', 'n_pos']]
    df = df.sort_values(['sdate', 'fdate'])

    df = df.loc[df['sdate'] >= pd.to_datetime(min_date)-pd.Timedelta(2, 'd')]

    _CACHE[(min_date, max_date)] = (time(), df.copy())
    return df


def plot_jump_10Aug():
    """Plot jump in data due to changes in counting around 10 Aug 2020."""

    df = load_testdata('2021-06-01', '2021-09-01')

    test_snapshots = {}
    for fdate in ['2021-08-09', '2021-08-11']:
        test_snapshots[fdate] = df.loc[df['fdate'] == fdate].set_index('sdate')

    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    # n = len(test_snapshots)
    for i, (fdate, df1) in enumerate(test_snapshots.items()):
        msize = (1.5-i)*40
        ax.scatter(df1.index, df1['n_tested'], label=f'Tests GGD, gepub. {fdate}',
                   color=colors[i], marker='o^'[i], s=msize)
        ax.scatter(df1.index, df1['n_pos'], label=f'Posi.  GGD, gepub. {fdate}',
                   color=colors[i], marker='+x'[i], s=msize*1.5)
    oneday = pd.Timedelta('1d')
    ax.set_xlim(df['sdate'].min()-oneday, df['sdate'].max()+oneday)
    ax.grid()
    ax.set_title('Aantal GGD tests per dag op twee publicatiedatums.')
    import tools
    tools.set_xaxis_dateformat(ax, xlabel='Datum monstername')
    ax.legend()
    fig.show()


def plot_daily_tests_and_delays(date_min, date_max='2099-01-01', src_col='n_tested'):
    """Note: date_min, date_max refer to file dates. Test dates are generally
    up to two days earlier. Repository 'mzelst-covid19' should be up to date.

    Optionally set src_col='n_pos'
    """

    # Convert to index 'sdate' and columns 2, 3, 4, ... with n_tested
    # at 2, 3, ... days after sampling date.
    df = load_testdata(date_min, date_max)

    n_days_wait = 5
    for iw in range(2, n_days_wait+1):
        df1 = df.loc[df['sdate'] == df['fdate'] - pd.Timedelta(iw, 'd')]
        df[iw] = df1[src_col]

    df = df.groupby('sdate').max()

    # Manually tweak around 2021-08-10 when counting changed.
    for iw in range(3, n_days_wait+1):
        sdate = pd.to_datetime('2021-08-10') - pd.Timedelta(iw, 'd')
        for j in range(iw, n_days_wait+1):
            df.at[sdate, j] = np.nan


    fig, ax = plt.subplots(figsize=(7.5, 5), tight_layout=True)
    barwidth = pd.Timedelta(1, 'd')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#dddddd', '#bbbbbb', '#999999', '#777777',
              '#555555', '#333333', '#111111'
              ] * 2

    for iw, color in zip(range(2, n_days_wait+1), colors):
        xs = df.index + pd.Timedelta(0.5, 'd')
        ytops = df[iw].values
        ybots = df[iw].values*0 if iw == 2 else df[iw-1].values
        edge_args = dict(
            edgecolor=('#444488' if len(xs) < 150 else 'none'),
            linewidth=min(1.5, 100/len(xs)-0.2),
            )
        ax.bar(xs, ytops-ybots, bottom=ybots, color=color,
               width=barwidth,
               label=(f'Na {iw} dagen' if iw < 10 else None),
               **edge_args
               )
        if iw == 2:
            xmask = df.index.dayofweek.isin([5,6])
            color = '#1b629a'
            ax.bar(
                xs[xmask], (ytops-ybots)[xmask], bottom=ybots[xmask], color=color,
                width=barwidth, **edge_args
               )

    if date_min < '2021-08-07' < date_max:
        ax.text(pd.to_datetime('2021-08-07T13:00'), 20500,
                'Telmethode gewijzigd!',
                rotation=90, horizontalalignment='center', fontsize=8
                )

    ytops = df.loc[:, range(2, n_days_wait+1)].max(axis=1, skipna=True)  # Series
    missing_day2 = (ytops - df[2])/ytops # Series
    missing_thresh = 0.005
    row_mask = (missing_day2 >= missing_thresh)
    for tm, y in ytops.loc[row_mask].items():
        if tm < pd.to_datetime(date_min):
            continue
        percent_late = np.around(missing_day2.loc[tm]*100, 1)
        ax.text(
            tm + pd.Timedelta(0.55, 'd'), y,
            f'  {percent_late:.2g}%', rotation=90,
            horizontalalignment='center', fontsize=8
            )

    ax.set_xlim(
        df1['sdate'].iloc[0] + pd.Timedelta(n_days_wait-1, 'd'),
        df1['sdate'].iloc[-1] + pd.Timedelta(n_days_wait-1, 'd')
        )
    if src_col == 'n_pos':
        ax.set_yscale('log')
        ax.set_ylim(1000, 20000)
        ax.set_title('Aantal positieve GGD-tests per dag')
        ax.set_xlabel('Datum monstername')
    else:
        ax.set_ylim(0, 1.15*df[src_col].max())
        ax.set_title('Aantal testuitslagen per dag - laatste drie dagen mogelijk incompleet')

    ax.set_xlabel('Datum monstername')

    ax.legend(loc='upper left')
    set_xaxis_dateformat(ax, maxticks=15)

    fig.show()


if __name__ == '__main__':

    plt.close('all')

    if 0:
        plot_jump_10Aug()

    plot_daily_tests_and_delays('2021-09-01')
    plot_daily_tests_and_delays('2021-09-01', src_col='n_pos')
