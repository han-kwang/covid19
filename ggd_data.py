#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Downloading and loading GGD test data.

Created on Sun Nov 28 20:34:13 2021

@author: @hk_nien
"""

from pathlib import Path
import urllib
import time
import re
import gzip
import numpy as np
import pandas as pd

# Look in these locations. Download into first location.
DATA_PATHS = [
    'data-rivm/tests',
    '../mzelst-covid19-nobackup/data-rivm/tests',
    ]

FNAME_TEMPLATE = 'rivm_daily_{date}.csv.gz'

def update_ggd_tests(force=False):
    """Update GGD test file in data-rivm/tests.

    Set force=True to force; otherwise decide automatically.
    """
    tm_now = pd.Timestamp('now')
    daytime = tm_now.strftime('%H:%M:%S')

    if daytime < '15:14:55':
        fdate = (tm_now - pd.Timedelta('17 h')).strftime('%Y-%m-%d')
    else:
        fdate = tm_now.strftime('%Y-%m-%d')


    fname = FNAME_TEMPLATE.format(date=fdate)
    fpath = Path(DATA_PATHS[0]) / fname
    if fpath.is_file() and not force:
        print(f'GGD test data already up to date: {fname}.')
        return

    if daytime > '15:14:55' and daytime < '15:15:20':
        print("It's exactly 15:15; waiting a few seconds for RIVM to update...",
              end='', flush=True)
        time.sleep((pd.to_datetime('15:15:21') - pd.to_datetime(daytime)).seconds)
        print('done.')


    url = 'https://data.rivm.nl/covid-19/COVID-19_uitgevoerde_testen.csv'
    print(f'Getting latest GGD test data from RIVM...')
    with urllib.request.urlopen(url) as response:
        csv_bytes = response.read()

    # Rows like this.
    # 2;2021-11-28 09:00:00;2020-06-01;VR01;Groningen;2;0
    # Should become like this (commas and timezone) for compatibility with
    # mzelst archive.
    # 2,2021-11-20 09:00:00Z,2020-06-01,VR01,Groningen,2,0

    lines = csv_bytes.decode('utf-8').splitlines()
    lines = [
        re.sub(r'00:00,', r'00:00Z,', li.replace(';', ','))
        for li in lines
        ]
    lines[-1] += '\n'
    txt = '\n'.join(lines)
    zdata = gzip.compress(txt.encode('utf-8'))
    tmp_fpath = Path(DATA_PATHS[0]) / f'{fname}.tmp'
    with tmp_fpath.open('wb') as f:
        f.write(zdata)
    tmp_fpath.rename(fpath)
    print(f'Wrote {fpath}')


def load_ggd_pos_tests(fdate=-1, quiet=False):
    """Get latest GGD test data as DataFrame.

    Parameters:

    - fdate: file date as str (yyyy-mm-dd) OR negative integer,
      with fdate=-1 referring to today's data; -2 to yesterday's, and so on.
      (before 15:15 local time, one day earlier).
    - quiet: True to suppress 'loaded ...' messages.

    Return DataFrame:

    - Index: Date_tested (timestamps at 12:00)
    - n_tested: number tested on that day (over entire country)
    - n_pos: number positive on that day

    Days with known bad data will have NaN.
    """
    if not isinstance(fdate, (str)):
        i_offs = int(fdate)
        tm_now = pd.Timestamp('now')
        daytime = tm_now.strftime('%H:%M')
        if daytime < '15:15':
            tm_now -= pd.Timedelta('20 h')
        fdate = (tm_now - pd.Timedelta(1+i_offs, 'd')).strftime('%Y-%m-%d')

    if not re.match('20\d\d-\d\d-\d\d', fdate):
        raise ValueError(f'Bad fdate: {fdate}')

    fname = FNAME_TEMPLATE.format(date=fdate)
    for data_path in DATA_PATHS:
        fpath = Path(data_path) / fname
        if fpath.is_file():
            break
    else:
        raise FileNotFoundError(f'{fname} in two directories. Run update_ggd_tests() first.')

    df = pd.read_csv(fpath).drop(columns='Version')
    if not quiet:
        print(f'Loaded {fpath}')
    df.rename(columns={'Date_of_statistics': 'Date_tested',
                       'Tested_with_result': 'n_tested',
                       'Tested_positive': 'n_pos'},
              inplace=True)
    df['Date_tested'] = pd.to_datetime(df['Date_tested'])
    df.drop(columns='Date_of_report', inplace=True)
    df = df.groupby('Date_tested').sum()

    bad_dates = ['2021-02-07']
    df.loc[pd.to_datetime(bad_dates), ['n_pos', 'n_tested']] = np.nan

    return df


# df = load_ggd_pos_tests()