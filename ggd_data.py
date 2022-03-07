#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Downloading and loading GGD test data.

Created on Sun Nov 28 20:34:13 2021

@author: @hk_nien
"""

from pathlib import Path
import urllib
import re
import gzip
import numpy as np
import pandas as pd
import tools

# Look in these locations. Download into first location.
DATA_PATHS = [
    'data-rivm/tests',
    '../mzelst-covid19-nobackup/data-rivm/tests',
    ]

FNAME_TEMPLATE = 'rivm_daily_{date}.csv.gz'
# Holiday regions (approximately)
REGIONS_NOORD = [
    'Amsterdam-Amstelland',
    'Drenthe',
    'Flevoland',
    'Fryslân',
    'Gooi en Vechtstreek',
    'Groningen',
    'IJsselland',
    'Kennemerland',
    'Noord- en Oost-Gelderland',
    'Noord-Holland-Noord',
    'Twente',
    'Zaanstreek-Waterland',
    ]
REGIONS_MIDDEN = [
    'Zuid-Holland-Zuid.'
    'Gelderland-Midden', # Includes Arnhem (which belongs to Zuid)
    'Haaglanden',
    'Hollands-Midden',
    'Rotterdam-Rijnmond',
    'Utrecht',
    ]
REGIONS_ZUID = [
    'Brabant-Noord',
    'Brabant-Zuidoost',
    'Gelderland-Zuid',
    'Limburg-Noord',
    'Limburg-Zuid',
    'Midden- en West-Brabant',
    'Zeeland',
    ]


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
    tools.wait_for_refresh('15:14:55', '15:15:45', 'Waiting until {t2} for GGD data')

    url = 'https://data.rivm.nl/covid-19/COVID-19_uitgevoerde_testen.csv'
    print('Getting latest GGD test data from RIVM...')
    with urllib.request.urlopen(url) as response:
        csv_bytes = response.read()

    # Rows like this.
    # 2;2021-11-28 09:00:00;2020-06-01;VR01;Groningen;2;0
    # Should become like this (commas and timezone) for compatibility with
    # mzelst archive.
    # 2,2021-11-20 09:00:00Z,2020-06-01,VR01,Groningen,2,0

    lines = csv_bytes.decode('utf-8').splitlines()

    # check that this is the date we want
    fdate_downloaded = lines[1].split(';')[1].split(' ')[0]
    if fdate_downloaded != fdate:
        tm_now = pd.Timestamp('now').strftime('%Y-%m-%d %H:%M:%S')
        raise ValueError(
            f'GGD data download: expected file date {fdate}, got {fdate_downloaded}. Now is {tm_now}.'
            )

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

_GGD_DF_CACHE = {}  # key: filename; value: full df copy


def load_ggd_pos_tests(fdate=-1, quiet=False, region_regexp=None):
    """Get latest GGD test data as DataFrame.

    Parameters:

    - fdate: file date as str (yyyy-mm-dd) OR negative integer,
      with fdate=-1 referring to today's data; -2 to yesterday's, and so on.
      (before 15:15 local time, one day earlier).
    - quiet: True to suppress 'loaded ...' messages.
    - region_regexp: optional regular expression to match against
      Security_region_name. Or HR:Zuid, HR:Noord, HR:Midden.

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
    if fname in _GGD_DF_CACHE:
        df = _GGD_DF_CACHE[fname].copy()
    else:
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

        # Update cache
        if len(_GGD_DF_CACHE) > 20:
            _GGD_DF_CACHE.clear()
        _GGD_DF_CACHE[fname] = df.copy()


    if region_regexp:
        if region_regexp.startswith('HR:'):
            rdict = {
                'Noord': REGIONS_NOORD,
                'Midden': REGIONS_MIDDEN,
                'Zuid': REGIONS_ZUID,
                }
            region_regexp = '|'.join(rdict[region_regexp[3:]])

        df1 = df.loc[df['Security_region_name'].str.contains(region_regexp)]
        if len(df1) == 0:
            known_regions = sorted(df['Security_region_name'].unique())
            print(f'Regions available: {", ".join(known_regions)}.')
            raise ValueError(f'No match of {region_regexp!r}.')
        else:
            df = df1

    df = df.groupby('Date_tested').sum()

    bad_dates = ['2021-02-07']
    df.loc[pd.to_datetime(bad_dates), ['n_pos', 'n_tested']] = np.nan

    return df


# df = load_ggd_pos_tests()