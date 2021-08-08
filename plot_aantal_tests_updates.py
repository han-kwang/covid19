#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 21:46:26 2021

@author: hk_nien @ Twitter
"""
from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tools import set_xaxis_dateformat


# Clone of the github.com/mzelst/covid-19 repo.
test_path = '../mzelst-covid19-nobackup/data-rivm/tests'

def load_testdata(min_date='2021-01-01'):
    """Return DataFrame. Specify mindate as 'yyyy-mm-dd'.

    Return DataFrame with multi-index (sdate, fdate).
    """

    fnames = sorted(Path(test_path).glob('rivm_daily_*.csv.gz'))
    dfs = []
    for fn in fnames:
        m = re.search(r'_(\d\d\d\d-\d\d-\d\d).csv.gz', str(fn))
        if not m:
            continue
        fdate = m.group(1)
        if fdate < min_date:
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

    return df


df = load_testdata('2021-06-15')
df_saved = df.copy()
#%%
# Convert to index 'sdate' and columns 2, 3, 4, ... with n_tested
# at 2, 3, ... days after sampling date.
df = df_saved.copy()
n_days_wait = 5
for iw in range(2, n_days_wait+1):
    df1 = df.loc[df['sdate'] == df['fdate'] - pd.Timedelta(iw, 'd')]
    df[iw] = df1['n_tested']

df = df.groupby('sdate').max()



plt.close('all')

fig, ax = plt.subplots(figsize=(7.5, 5), tight_layout=True)
barwidth = pd.Timedelta(1, 'd')


for iw in range(2, n_days_wait+1):
    xs = df.index + pd.Timedelta(0.5, 'd')
    ytops = df[iw].values
    ybots = df[iw].values*0 if iw == 2 else df[iw-1].values
    ax.bar(xs, ytops-ybots, bottom=ybots,
           width=barwidth*0.9, label=f'Na {iw} dagen')

ytops = df.loc[:, range(2, n_days_wait+1)].max(axis=1, skipna=True)  # Series
missing_day2 = (ytops - df[2])/ytops # Series
missing_thresh = 0.005
row_mask = (missing_day2 >= missing_thresh)
for tm, y in ytops.loc[row_mask].items():
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
ax.set_ylim(0, 1.15*df['n_tested'].max())

ax.set_xlabel('Datum monstername')

ax.legend(loc='upper left')
set_xaxis_dateformat(ax, maxticks=15)
ax.set_title('Aantal testuitslagen per dag - laatste drie dagen mogelijk incompleet')

fig.show()

