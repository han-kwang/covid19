#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Covid statistics Netherlands - functions related to data loading.

Exports:

- `init_data()`: load data, store into global DFS, download updates if necessary.
- `DFS`: dict with initialized data (DataFrames).


Created on Sat Oct 23 16:06:32 2021

@author: @hk_nien on Twitter
"""
import re
import io
import urllib
import urllib.request
from pathlib import Path
import time
import pandas as pd
import nl_regions


try:
    DATA_PATH = Path(__file__).parent / 'data'
except NameError:
    DATA_PATH = Path('data')

# this will contain dataframes, initialized by init_data().
# - mun: municipality demograhpics
# - cases: cases by municipality
# - events: Dutch events by date
# - Rt_rivm: RIVM Rt estimates
# - anomalies: anomaly data
DFS = {}



def _str_datetime(t):
    return t.strftime('%a %d %b %H:%M')


def load_events():
    """Return events DataFrame.

    - index: DateTime start.
    - 'Date_stop': DateTime
    - 'Description': Description string.
    - 'Flags':  None or string (to indicate that it is only shown in a
      particular type of grahp.
    """

    df = pd.read_csv(DATA_PATH / 'events_nl.csv', comment='#')
    df['Date'] = pd.to_datetime(df['Date']) + pd.Timedelta('12:00:00')
    df['Date_stop'] = pd.to_datetime(df['Date_stop'])
    df.loc[df['Flags'].isna(), 'Flags'] = None
    df.set_index('Date', inplace=True)
    return df


def download_Rt_rivm_coronawatchNL(maxage='16 days 15 hours',  force=False):
    """Download reproduction number from RIVM if new version is available.

    Old function; CoronawatchNL is not updated anymore, so it seems
    (2020-12-20).

    Parameters:

    - maxage: maximum time difference between last datapoint and present time.
    - force: whether to download without checking the date.

    Usually, data is published on Tue 14:30 covering data up to Sunday the
    week before, so the data will be stale after 16 days 14:30 h.
    Data is downloaded from a Git repository that has some delay w.r.t. the
    RIVM publication.
    """

    # get last date available locally
    fpath = Path('data/RIVM_NL_reproduction_index.csv')
    if fpath.is_file():
        df = pd.read_csv(fpath)
        last_time = pd.to_datetime(df['Datum'].iloc[-1])
        now_time = pd.to_datetime('now')
        if not (force or now_time >= last_time + pd.Timedelta(maxage)):
            print('Not updating RIVM Rt data; seems recent enough.')
            return
        local_file_data = fpath.read_bytes()
    else:
        local_file_data = b'dummy'

    # Data via CoronawatchNL seems stale as of 2020-12-20
    url = 'https://raw.githubusercontent.com/J535D165/CoronaWatchNL/master/data-dashboard/data-reproduction/RIVM_NL_reproduction_index.csv'
    print(f'Getting latest R data ...')
    with urllib.request.urlopen(url) as response:
        data_bytes = response.read()
        if data_bytes == local_file_data:
            print(f'{fpath}: already latest version.')
        else:
            fpath.write_bytes(data_bytes)
            print(f'Wrote {fpath} .')

def download_Rt_rivm(force=False):
    """Download reproduction number from RIVM if new version is available.

    Parameters:

    - force: whether to download without checking the date.

    Usually, data is published on Fri and Tue, 15:15 or 15:25.

    For history purposes, files will be saved as

    'rivm_reproductiegetal.csv' (latest)
    'rivm_reproductiegetal-yyyy-mm-dd.csv' (by most recent date in the data file).

    The file 'rivm_R_updates.csv' will be updated as well.
    """

    # get last date available locally
    fname_tpl = 'data-rivm/R_number/rivm_reproductiegetal{}.csv'
    fpath = Path(fname_tpl.format(''))

    if fpath.is_file():
        df = pd.read_csv(fpath)
        last_time = pd.to_datetime(df['Date'].iloc[-1])
        # File release date 5 days after last datapoint.
        last_release_time = last_time + pd.Timedelta('1 days, 15:15:00')
        # release day of week -> time to next release
        release_dows = {1: 3, 4: 4}
        try:
            days_to_next = release_dows[last_release_time.dayofweek]
        except KeyError:
            print(f'Warning: please check {__file__}: download_Rt_Rivm().\n'
                  'Release schedule seems to have changed.')
            days_to_next = 1
        next_release_time = last_release_time + pd.Timedelta(days_to_next, 'd')
        now_time = pd.Timestamp.now()  # this will be in local time
        if not force and now_time < next_release_time:
            print('Not updating RIVM Rt data; seems recent enough')
            print(f'  Inferred last release: {_str_datetime(last_release_time)}; '
                  f'now: {_str_datetime(now_time)}; next release: {_str_datetime(next_release_time)}.')
            return
        local_file_data = fpath.read_bytes()
    else:
        local_file_data = b'dummy'

    url = 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'
    print(f'Getting latest R data from RIVM...')
    with urllib.request.urlopen(url) as response:
        json_bytes = response.read()
        f = io.BytesIO(json_bytes)

    df = pd.read_json(f) # Columns Date, Rt_low, Rt_avg Rt_up, ...
    df.set_index('Date', inplace=True)

    f = io.StringIO()
    df.to_csv(f, float_format='%.4g')
    f.seek(0)
    data_bytes = f.read().encode('utf-8')

    if data_bytes == local_file_data:
        print(f'{fpath}: already latest version.')
        return

    # New dataset! Write it.
    fpath.write_bytes(data_bytes)
    print(f'Wrote {fpath} .')
    ymd = df.index[-1].strftime('-%Y-%m-%d')
    fpath_arch = Path(fname_tpl.format(ymd))
    fpath_arch.write_bytes(data_bytes)
    print(f'Wrote {fpath_arch} .')

    # and update updates file with latest defined Rt_avg value.
    upd_fname = 'data/rivm_R_updates.csv'
    upd_df = pd.read_csv(upd_fname).set_index('Date') # column 'Rt_update'
    # Compare to the last valid Rt record.
    df_latest = df.loc[~df['Rt_avg'].isna()].iloc[-1:][['Rt_avg']].rename(columns={'Rt_avg': 'Rt_update'})
    df_latest.index = [df_latest.index[0].strftime('%Y-%m-%d')]
    df_latest.index.name ='Date'
    print(f'df_latest:\n{df_latest}')
    print(f'upd_df:\n{upd_df.iloc[-5:]}')

    print(f'compare: {upd_df.index[-1]!r} {df_latest.index[0]!r}')

    if upd_df.index[-1] < df_latest.index[0]:
        upd_df = upd_df.append(df_latest)
        upd_df.to_csv(upd_fname)
        print(f'Updated {upd_fname} .')



def load_rivm_R_updates(dates=None):
    """Load data/rivm_R_updates.csv.

    Optionally select only rows present in 'dates' (should be pd.DateTime
    at 12:00h on the day).

    Return Series with 'Rt_update' name.
    """
    df = pd.read_csv('data/rivm_R_updates.csv')
    assert list(df.columns) == ['Date', 'Rt_update']
    df['Date'] = pd.to_datetime(df['Date']) + pd.Timedelta(12, 'h')
    df = df.set_index('Date')

    if dates is not None:
        df = df.loc[df.index.isin(dates)]
        if len(df) == 0:
            print('Warning: load_rivm_R_updates - no date match.')
    return df


def load_Rt_rivm(autoupdate=True, source='rivm'):
    """Return Rt DataFrame, with Date index (12:00), columns R, Rmax, Rmin.

    Source can be 'rivm' or 'coronawatchnl'.
    """

    if source == 'coronawatchnl':
        if autoupdate:
            download_Rt_rivm_coronawatchNL()
        df_full = pd.read_csv('data/RIVM_NL_reproduction_index.csv')
        df_full['Datum'] = pd.to_datetime(df_full['Datum']) + pd.Timedelta(12, 'h')
        df = df_full[df_full['Type'] == ('Reproductie index')][['Datum', 'Waarde']].copy()
        df.set_index('Datum', inplace=True)
        df.rename(columns={'Waarde': 'R'}, inplace=True)
        df['Rmax'] = df_full[df_full['Type'] == ('Maximum')][['Datum', 'Waarde']].set_index('Datum')
        df['Rmin'] = df_full[df_full['Type'] == ('Minimum')][['Datum', 'Waarde']].set_index('Datum')
        df['Rt_update'] = load_rivm_R_updates(df.index)
        return df

    if source == 'rivm':
        if autoupdate:
            download_Rt_rivm()

        df = pd.read_csv('data-rivm/R_number/rivm_reproductiegetal.csv')

        df2 = pd.DataFrame(
            {'Datum': pd.to_datetime(df['Date']) + pd.Timedelta(12, 'h')}
            )
        df2['R'] = df['Rt_avg']
        df2['Rmin'] = df['Rt_low']
        df2['Rmax'] = df['Rt_up']
        df2.set_index('Datum', inplace=True)

        # last row may be all-NaN; eliminate such rows.
        df2 = df2.loc[~df2['Rmin'].isna()]
        df2['Rt_update'] = load_rivm_R_updates(df2.index)
        return df2


    raise ValueError(f'source={source!r}')


def check_RIVM_message():
    """Check for messages on data problems on the RIVM page. Print warning if so."""

    url = 'https://data.rivm.nl/covid-19/'
    with urllib.request.urlopen(url) as response:
        data_bytes = response.read()
        # Page charset according to server is ISO-8859-1, but it
        # contained character \x92 (windows-1252 apostroph) on 2021-11-16.
        htm = data_bytes.decode('windows-1252')

    # Trim html response
    htm = re.sub('<pre>.*$', '', htm, flags=re.S)  # pre starts the list of files.
    # HTML formatting of this block changes every time, but it's after the
    # "daily 15:15" message.
    # Remove comment (warning message stored in comment), other non-text stuff
    delete_regexps = [
        '<!--.*?-->',
        '<[^>]+>',
        '.*15 hours.'
       ]

    for regex in delete_regexps:
        htm = re.sub(regex, '', htm, flags=re.S)
    htm = re.sub(r'(\s)\s*', r'\1', htm)

    if len(htm) > 500:
        htm = f'{htm[:497]}...'
    if re.search('storing|onderraportage|achterstand', htm):
        print(f'Warning: RIVM data page says:\n{htm}')


def get_municipalities_by_pop(minpop, maxpop, sort='size'):
    """Return list of municipalities with populations in specified range.

    sort by 'size' or 'alpha'.
    """

    df = DFS['mun']
    pops = df['Population']
    df = df.loc[(pops >= minpop) & (pops < maxpop)]

    if sort == 'alpha':
        return sorted(df.index)

    if sort == 'size':
        return list(df.sort_values(by='Population', ascending=False).index)

    raise ValueError(f'sort={sort!r}')



def update_cum_cases_csv(force=False):
    """Update 'cumulative data' csv file (if not recently updated)."""

    fpath = DATA_PATH / 'COVID-19_aantallen_gemeente_cumulatief.csv'
    if fpath.is_file():
        local_file_data = fpath.read_bytes()
    else:
        local_file_data = None

    if not force:
        if fpath.is_file():
            # estimated last update
            tm = time.time()
            loc_time = time.localtime(tm)
            day_seconds = loc_time[3]*3600 + loc_time[4]*60 + loc_time[5]
            # RIVM releases data at 15:15.
            tm_latest = tm - day_seconds + 15.25*3600
            if tm_latest > tm:
                tm_latest -= 86400

            tm_file = fpath.stat().st_mtime
            if tm_file > tm_latest + 60: # after 15:15
                print('Not updating cumulative case file; seems to be recent enough.')
                return
            if tm_file > tm_latest:
                print('Cumulative case data file may or may not be the latest version.')
                print('Use update_cum_cases_csv(force=True) to be sure.')
                return

    #check_RIVM_message()

    url = 'https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_cumulatief.csv'
    print(f'Getting new daily case statistics file...')
    with urllib.request.urlopen(url) as response:
        data_bytes = response.read()
        if data_bytes == local_file_data:
            print(f'{fpath}: already latest version.')
        else:
            fpath.write_bytes(data_bytes)
            print(f'Wrote {fpath} .')


def load_cumulative_cases(autoupdate=True):
    """Return df with cumulative cases by municipality. Retrieve from internet if needed.

    - autoupdate: whether to retrieve latest data from internet.
    """

    if autoupdate:
        update_cum_cases_csv()

    df = pd.read_csv('data/COVID-19_aantallen_gemeente_cumulatief.csv', sep=';')
    # Removing 'municipality unknown' records.
    # Not clear; including these, the daily numbers are higher than the official
    # report. With them removed, they are lower.
    # Examples:
    # date: incl NA, w/o NA, nos.nl, allecijfers.nl
    # 2020-11-08: 5807 5651 5703 5664
    # 2020-11-06: 7638 7206 7272 7242
    df = df.loc[~df.Municipality_code.isna()] # Remove NA records.
    df['Date_of_report'] = pd.to_datetime(df['Date_of_report'])

    df.loc[df['Municipality_name'] == 'Hengelo (O.)', 'Municipality_name'] = 'Hengelo'

    return df



def init_data(autoupdate=True, Rt_source='rivm'):
    """Init global dict DFS with 'mun', 'Rt_rivm', 'cases', 'events'.

    Parameters:

    - autoupdate: whether to attempt to receive recent data from online
      sources.
    - Rt_source: 'rivm' or 'coronawatchnl' (where to download Rt data).
    """

    DFS['cases'] = df = load_cumulative_cases(autoupdate=autoupdate)
    DFS['mun'] = nl_regions.get_municipality_data()
    DFS['Rt_rivm'] = load_Rt_rivm(autoupdate=autoupdate, source=Rt_source)
    DFS['events'] = load_events()
    dfa = pd.read_csv('data/daily_numbers_anomalies.csv', comment='#')
    dfa['Date_report'] = pd.to_datetime(dfa['Date_report']) + pd.Timedelta('10 h')
    DFS['anomalies'] = dfa.set_index('Date_report')

    print(f'Case data most recent date: {df["Date_of_report"].iat[-1]}')
    # get_mun_data('Nederland', 5e6, printrows=5)

