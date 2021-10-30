#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for casus analysis. For importing.

Functions:

- set_conf(): set global parameters (paths, cpu usage).
- download_rivm_casus_files(): download casus file (via github.com/mzelst)
- load_casus_data(): Return DataFrame with casus data from one RIVM csv file.
- load_casus_summary(): Load casus data and summarize into dataframe.
- load_merged_summary(): load combined summaries over range of file dates,
  from csv file as created by create_merged_summary_csv()
- create_merged_summary_csv(): create csv file with merged summary data.
- add_eDOO_to_summary(): add eDOO (estimated DOO) column to summary data.
- add_deltas_to_summary(): add changes in DOO etc. from file date to file date.
  (this is not very useful.)
- save_data_cache(): save a dataframe or other structure to cache file
- load_data_cache(): load a dataframe or othor structure from cache file
- PoolNCPU: multiprocessing.Pool wrapper.

Classes:

- DOOCorrection: correction factors for recent daily eDOO values.

Dataframes typically have subsets of these indices and columns:

   Indices:

   - Date_file: release date of underlying file (time 0:00 implied)
   - Date_statistics: date corresponding to the statistics (snapshot for
     a given Date_file)

   Columns:

    - DON: total 'date of notification' for this date
    - DOO: total 'date of disease onset' for this date
    - DPL: total 'date of positive lab result' for this date
    - Dtot: total sum of the above three columns
    - eDOO: estimated DOO total (by adding shifted DON, DPL data,
      from the same file date).
    - sDON: shifted/blurred DON number
    - sDPL: shifted/blurred DPL number.

Created on Sun Nov  8 22:44:09 2020

@author: @hk_nien
"""

import re
import locale
import pickle
import time
import io
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import urllib.request
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal
import tools

# Note: need to run this twice before NL locale takes effect.
try:
    locale.setlocale(locale.LC_ALL, 'nl_NL.UTF-8')
except locale.Error as e:
    print(f'Warning: cannot set language: {e.args[0]}')






CONFIG = dict(
    # casus data (huge gzipped csv files)
    cdpath=Path(__file__).parent / 'data-casus',
    # other data
    dpath=Path(__file__).parent / 'data',
    # other data
    cache_path=Path(__file__).parent / 'cache',
    # managing multiprocessing
    max_num_cpu=6, # max number of cpus
    max_frac_cpu=0.75, # max fraction of available cpus to use.
    # list of directories that may contain a clone of the repository
    # https://github.com/mzelst/covid-19.git .
    # For getting the latest 'casus data', it will check there before
    # downloading from internet.
    # parents[1] means one directory up from the script directory.
    mzelst_repro_dirs = [
        Path(__file__).parents[1] / 'mzelst-covid19-nobackup',
        Path(__file__).parents[1] / 'mzelst-covid-19',
        ],
    )


def set_conf(**kwargs):
    """Set configuration parameters (CONFIG global variable)."""

    for k, v in kwargs.items():
        if k in CONFIG:
            CONFIG[k] = type(CONFIG[k])(v)
        else:
            raise KeyError(k)

def PoolNCPU(msg=None):
    """Return Pool reasonable number of CPUs to use.

    Optional message that may include {ncpu}.
    If ncpu=1, return a dummy object that does an unpooled map.
    """
    # Load data with multi CPU, don't eat up all CPU
    ncpu = max(1, int(cpu_count() * CONFIG['max_frac_cpu']))
    ncpu = min(CONFIG['max_num_cpu'], ncpu)
    if msg:
        print(msg.format(ncpu=ncpu))

    class DummyPool:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def map(self, f, inputs):
            return map(f, inputs)

    if ncpu > 1:
        return Pool(ncpu)

    return DummyPool()


def _get_mzelst_casus_data_path_list():
    """Return list with 0 or 1 Path entries to mzelst casus data dir."""

    for p in CONFIG['mzelst_repro_dirs']:
        p = p / 'data-rivm/casus-datasets'
        if p.is_dir():
            return [p]

    return []

def _find_casus_fpath(date):
    """Return COVID-19_casus_landelijk_xxx.csv.gz file path for this date (yyyy-mm-dd)."""


    dirpaths = [CONFIG['cdpath']] + _get_mzelst_casus_data_path_list()
    fname = f'COVID-19_casus_landelijk_{date}.csv.gz'
    for dirpath in dirpaths:
        fpath = dirpath / fname
        if fpath.exists():
            return fpath
    else:
        raise FileNotFoundError(f'{fname} in {len(dirpaths)} directories.')


def load_casus_data(date):
    """Return DataFrame with casus data from one RIVM csv file.

    Parameter:

    - date: 'yyyy-mm-dd' string or timestamp

    Load from data-casus/COVID-19_casus_landelijk_xxx.csv.gz
    Date_file and Date_statistics will be timestamps (at time 0:00)
    """
    if not isinstance(date, str):
        date = date.strftime('%Y-%m-%d')

    fpath = _find_casus_fpath(date)

    df = pd.read_csv(fpath, sep=',')

    # Convert 'yyyy-mm-dd 10:00' to a 'yyyy-mm-dd' timestamp, because the 10:00
    # is just clutter when comparing dates.
    ymd = df['Date_file'].str.extract(r'(....-..-..)', expand=False)
    df['Date_file'] = pd.to_datetime(ymd)
    df['Date_statistics'] = pd.to_datetime(df['Date_statistics'])
    return df

def load_casus_summary(date):
    """Return summary dataframe for this date.

    This will reload an RIVM casus dataset. This is slow (few seconds
    for dates in 2021).

    Date format: 'yyyy-mm-dd'.

    Return DataFrame colomns:

    - Date_statistics (index)
    - Date_file
    - Dtot
    - DON
    - DOO
    - DPL
    """
    df = load_casus_data(date)

    tstats = df['Date_statistics'].unique()
    dfsum = pd.DataFrame(index=tstats)
    dfsum.index.name = 'Date_statistics'
    dfsum['Date_file'] = df['Date_file'].iat[0]
    for stype in ['DON', 'DOO', 'DPL']:
        df_select = df[df['Date_statistics_type'] == stype]
        dfsum[stype] = df_select.groupby('Date_statistics')['Date_file'].count()
        dfsum.loc[dfsum[stype].isna(), stype] = 0
        dfsum[stype] = dfsum[stype].astype(np.int32)


    dfsum['Dtot'] = dfsum['DOO'] + dfsum['DON'] + dfsum['DPL']

    return dfsum



def download_rivm_casus_files(force_today=False):
    """Download missing day files in data-casus.

    Download from here:
    https://github.com/mzelst/covid-19/tree/master/data-rivm/casus-datasets
    or from a local clone of that repository (as specified in the global CONFIG
    variable)

    The CSV layout is slightsly different from the RIVM data.
    The files as downloaded are huge, though highly compressible and stored
    here in gzipped format. To download all files rather than a few
    incrementally, it may be better to download a repository ZIP file from
    Github.

    files /COVID-19_casus_landelijk_yyyy-mm-dd.csv.gz.

    Return: number of files downloaded.
    """

    fdates = set()
    cdata_paths = [CONFIG['cdpath']] + _get_mzelst_casus_data_path_list()

    for cdata_path in cdata_paths:
       for fn in cdata_path.glob('COVID-19_casus_landelijk_*.csv.gz'):
            fn = str(fn)
            ma = re.search(r'(\d\d\d\d-\d\d-\d\d).csv.gz$', fn)
            if not ma:
                print(f'Pattern mismatch on {fn}.')
                continue
            fdates.add(ma.group(1))

    # start date
    tm_start = pd.to_datetime('2020-07-01') # first date in repository

    # end date is now or yesterday depending on time of day.
    tm_now = pd.to_datetime('now')
    if tm_now.hour < 15 and not force_today:
        tm_now -= pd.Timedelta('1 d')
    tm_end = pd.to_datetime(tm_now.strftime('%Y-%m-%d'))

    # which stuff to fetch...
    fdates_missing = []
    tm = tm_start
    while tm <= tm_end:
        ymd = tm.strftime('%Y-%m-%d')
        if ymd not in fdates:
            fdates_missing.append(ymd)
        tm += pd.Timedelta(1, 'd')

    if len(fdates_missing) == 0:
        print('Casus files up to date.')
        return 0

    if len(fdates_missing) > 10:
        input(f'Warning: do you really want to download {len(fdates_missing)}'
              ' huge data files? Ctrl-C to abort, ENTER to continue.')
    else:
        print(f'Will attempt to download case data for {len(fdates_missing)} days.')

    fname_template = 'COVID-19_casus_landelijk_{date}.csv.gz'
    url_template = (
        'https://github.com/mzelst/covid-19/raw/master/data-rivm/'
        f'casus-datasets/{fname_template}')

    for fdate in fdates_missing:
        url = url_template.format(date=fdate)
        print(f'Downloading casus data for {fdate} ...')
        fpath = CONFIG["cdpath"] / (fname_template.format(date=fdate))
        with urllib.request.urlopen(url) as response:
            data_bytes = response.read()
            if response.code != 200:
                raise FileNotFoundError(f'Response {response.code} on {url}')
            with gzip.open(io.BytesIO(data_bytes), 'rb') as f:
                data_bytes_unzip = f.read()
            if ('Date_statistics' not in data_bytes_unzip[:100].decode('utf-8')
                or fdate not in data_bytes_unzip[-200:].decode('utf-8')):
                # RIVM website for one does not give a 404 when it should
                raise ValueError(f'Bad or incomplete data in {url}.')

        with fpath.open('wb') as f:
            f.write(data_bytes)

        print(f'Wrote {fpath} .')

    return len(fdates_missing)


def _load_one_df(date):
    """Helper function for load_merged_summary in multiproc pool."""

    # print(f'({date}) ', end='', flush=True)
    print('.', end='', flush=True)
    return load_casus_summary(date).reset_index()

def load_merged_summary(date_lo, date_hi, reprocess='auto'):
    """Return merged summary DataFrame between two yyyy-mm-dd file dates.

    Parameters:

    - date_lo, date_hi: lowest, highest date_file ('yyyy-mm-dd').
      (no hours/minutes)
    - reprocess: True to reprocess all raw data (slow), False to use only
      preprocessed data (data/casus_history_summary.csv),
      'auto' to combine preexisting data if available.

    Return: DataFrame with columns: Date_statistics, Date_file, DON, DOO, DPL,
    Dtot.
    """
    assert reprocess in [True, False, 'auto']

    if reprocess in (False, 'auto'):
        dfs_old = load_merged_summary_csv(date_lo, date_hi)
        # multiindex -> columns
        dfs_old.reset_index(inplace=True)
        n_old_file = len(dfs_old['Date_file'].unique())
        print(f'Loaded already-preprocessed {n_old_file} file dates.')
        if not reprocess:
            return dfs_old

    if reprocess == 'auto' and len(dfs_old) > 0:
        date_lo = dfs_old['Date_file'].max() + pd.Timedelta(1, 'd')

    # Build list of file dates to read
    date_lo = pd.to_datetime(date_lo)
    date_hi = pd.to_datetime(date_hi)
    date = date_lo
    dfsums = []
    fdates = []
    while date <= date_hi:
        date_str = date.strftime('%Y-%m-%d')
        try:
            _find_casus_fpath(date_str)
        except FileNotFoundError:
            print(f'Using casus data before {date_str}.')
            break
        fdates.append(date_str)
        date += pd.Timedelta(1, 'd')

    if len(fdates) > 0:
        msg = f'Loading casus data for {len(fdates)} days (using {{ncpu}} processes)'
        with PoolNCPU(msg) as pool:
            dfsums = pool.map(_load_one_df, fdates)
            print()
    if reprocess == 'auto':
        dfsums.insert(0, dfs_old)

    if len(dfsums) == 0:
        raise ValueError('No data.')
    dfsmerged = pd.concat(dfsums)

    return dfsmerged

def create_merged_summary_csv(date_lo='2020-07-01', reprocess='auto'):
    """Load lots fo data, write to data/casus_history_summary.csv.

    See load_merged_sumarray() for column layout.
    """

    dfmerged = load_merged_summary(date_lo, '2099-01-01')
    fpath = CONFIG["dpath"] / 'casus_history_summary.csv'
    dfmerged.to_csv(fpath)
    print(f'Wrote {fpath} .')

def load_merged_summary_csv(date_lo='2020-07-01', date_hi='2099-01-01'):
    """Return history summary dataframe as stored in CSV file.

    Dataframe layout:

    Multiindex: Date_file, Date_statistics

    Columns:

    - DON: total 'date of notification' for this date
    - DOO: total 'date of disease onset' for this date
    - DPL: total 'date of positive lab result' for this date
    - Dtot: total sum of the above three columns
    - eDOO: estimated DOO total (by adding shifted DON, DPL data,
      from the same file date).
    - sDON: shifted/blurred DON number
    - sDPL: shifted/blurred DPL number.
    """

    fpath = CONFIG["dpath"] / 'casus_history_summary.csv'
    df = pd.read_csv(fpath, index_col=0)
    for col in ['Date_statistics', 'Date_file']:
        df[col] = pd.to_datetime(df[col])

    df.set_index(['Date_file', 'Date_statistics'], inplace=True)

    dtfs = df.index.get_level_values('Date_file').unique().sort_values()
    date_lo = pd.to_datetime(date_lo)
    date_hi = pd.to_datetime(date_hi)

    date_select = dtfs[(dtfs >= date_lo) & (dtfs <= date_hi)]
    df = df.loc[date_select]
    return df


def _add_eDOO_to_df1(args):
    """Helper for add_eDOO_to_summary().

    args: (dtf, df1, delay_don, delay_dpl, blur).

    Return (dtf, updated_df1)
    """
    (dtf, df1, delay_don, delay_dpl, blur) = args
    print('.', end='', flush=True)

    for col, delay in [('DON', delay_don), ('DPL', delay_dpl)]:
        df2 = df1[col].reset_index() # columns 'Date_statistics'
        df2['Date_statistics'] -= pd.Timedelta(delay, 'd')

        # df1a: shifted dataframe, index Date_statistics, column col.
        df1a = df1[[]].merge(df2, 'left', left_index=True, right_on='Date_statistics')
        df1a.set_index('Date_statistics', inplace=True)
        df1a[df1a[col].isna()] = 0
        if blur > 0 and len(df1a) > (2*blur + 1):
            kernel = np.full(2*blur+1, np.around(1/(2*blur+1), 1 + (blur > 2)))
            kernel[blur] += 1 - kernel.sum()
            df1a[col] = np.convolve(df1a[col], kernel, mode='same')
        df1[f's{col}'] = df1a[col]
        df1['eDOO'] += df1a[col]
    df1['Date_file'] = dtf
    df1 = df1.reset_index().set_index(['Date_file', 'Date_statistics'])
    return dtf, df1


def add_eDOO_to_summary(df, delay_don=3, delay_dpl=2, blur=1):
    """Add eDOO column - estimated DOO - to summary dataframe.

    Also add sDON and sDPL columns, which are estimated shifted
    DON/DPL values (with blurring).

    Parameters:

    - delay_don: assumed delay (days) from date of notification -> day of onset.
    - delay_dpl: assumed delay from date of positive lab result -> day of onset.
    - blur: additional blurring (half-width integer)
    """

    # create new columns
    df['eDOO'] = df['DOO'].copy()
    df['sDON'] = 0
    df['sDPL'] = 0

    file_dates = df.index.get_level_values('Date_file').unique().sort_values()
    # List of tuples (fdate, df_slice)
    map_input = [
        (dtf, df.loc[dtf].copy(), delay_don, delay_dpl, blur) # dataframe for one file, index Date_statistics
        for dtf in file_dates
        ]
    cols = ['eDOO', 'sDON', 'sDPL']

    msg = f'Adding eDOO for {len(map_input)} days ({{ncpu}} processes)'
    with PoolNCPU(msg) as pool:
        map_output = pool.map(_add_eDOO_to_df1, map_input)

    print('\nMerging...', end='', flush=True)
    for (dtf, df1) in map_output:
        df.loc[(dtf,), cols] = df1[cols]
    print('done.')
    return df


def add_deltas_to_summary(df):
    """Add delta columns dDOO, dDPL, dDON, dDtot to summary dataframe."""

    # file dates for deltas (except the first one).
    dtfs = df.index.get_level_values('Date_file').unique().sort_values()[1:]
    # create new columns
    for col in ['dDOO', 'dDON', 'dDPL', 'dDtot']:
        df[col] = 0.0

    print('Getting deltas by file date', end='')
    for dtf in dtfs:
        print('.', flush=True, end='')
        dtf_prev = dtf - pd.Timedelta(1, 'd')
        dfcurr = df.loc[(dtf,)] # dataframe with only Date_statistics as index
        dfprev = df.loc[(dtf_prev,)]
        deltas = {}
        for col in ['DOO', 'DPL', 'DON', 'Dtot']:
            dcol = f'd{col}'
            deltas[dcol] = dfcurr[col] - dfprev[col]
            if dtf in dfcurr.index:
                last_entry = dfcurr.loc[dtf, col] # there is no previous
            else:
                last_entry = 0
            deltas[dcol].loc[dtf] = last_entry

        df_deltas = pd.DataFrame(deltas)
        # add index
        df_deltas['Date_file'] = dtf
        df_deltas.reset_index(inplace=True)
        df_deltas.set_index(['Date_file', 'Date_statistics'], inplace=True)
        df.loc[(dtf,), ['dDOO', 'dDON', 'dDPL', 'dDtot']] = df_deltas

    print('')
    return df


def _ymd(date):
    return date.strftime('%Y-%m-%d')


def _show_save_fig(fig, show, fname):
    """Optionally show plot, optionaly save figure.
    Delete figure if no show.
    """

    if show:
        fig.show()

    if fname:
        fig.savefig(str(fname))
        print(f'Wrote {fname} .')

    if not show:
        plt.close(fig)


class DOOCorrection:
    """Coverage of recent DOO reports.

    For a stat_date that is j days before the file_date, G[j]
    is the estimated fraction of the true number of DOO cases for
    that stat_date. This calculation method was proposed by @bslagter
    on Twitter.

    Furthermore, two day-of-week (DoW) corrections are available:

    - DoW correction for file date: reports released on Sundays have
      different correction factors from those released on Mondays.
    - DoW correction for disease-onset (DOO) date. More people report Monday as
      the first day compared to Sunday.

    To initialize the 'G' correction, use from_doo_df(). To get the data for
    one full month, provide the end date 18 days into the next month.

    For the DoW corrections, data based on a single month is probably to noisy
    to see DoW effects clearly. therefore, the workflow is:

    1. use calc_fdow_correction() over a longer period to get f_dow_corr.
    2. use calc_sdow_correction() over a longer period to get s_dow_ce.
    3. use from_doo_df() over e.g. 1 month to get a full DOOCorrection.

    Attributes:

    - G: array (m,) with G values
    - eG: array (m,) with estimated errors in G
    - date_range: [date_lo, date_hi] (file dates that the correction
      was evaluated for. The input file dates extend m more days.)
    - f_dow_corr: file-date DoW correction factors for G, shape (7, m).
      (use G*f_dow_corr[dow] or iG/f_dow_corr[dow])
    - s_dow_corr: statistics-date DoW correction factors for nDOO, shape (7,)
      (use iG*s_dow_corr)
    - s_dow_corr_err: standard error on s_dow_corr (float)

    - iG: inverse G
    - eiG: standard error on iG

    Functions:

    - __init__(): direct initialization of all attributes.
    - from_doo_df(): analyze casus dataframe.
    - create_df_nDOO(self, df): create new dataframe with estimated nDOO.
    - calc_sdow_correction(): calculate stat-date DoW correction factor.
    - calc_fdow_correction(): calculate file-date DoW correction factors.
    - plot(): plot the correction curve.
    - plot_fdow_corr(): plot the file-date DoW correction curve.
    - plot_nDOO(): plot nDOO data (from eDOO column in dataframe)
    """

    def __init__(self, G, eG, date_range, f_dow_corr, s_dow_ce):
        """Init attributes directly.

        f_dow_ce, s_dow_ce are tuples (*_corr, *_corr_err)
        """

        self.G = np.array(G)
        self.eG = np.array(eG)
        self.date_range = [pd.to_datetime(x) for x in date_range[:2]]

        self.iG = 1/np.where(self.G!=0, self.G, np.nan)
        mask_big_err = (self.G < 2*self.eG)
        mask_big_iG = (np.where(np.isfinite(self.iG), self.iG, 999)  > 5)

        self.iG[mask_big_err | mask_big_iG] = np.nan
        self.eiG = self.eG*self.iG**2

        self.f_dow_corr = np.array(f_dow_corr)
        self.s_dow_corr = np.array(s_dow_ce[0])
        self.s_dow_corr_err = float(s_dow_ce[1])

    def __repr__(self):
        cname = self.__class__.__name__
        has_s_dowcorr = np.any(self.s_dow_corr != 1)
        has_f_dowcorr = np.any(self.f_dow_corr != 1)
        dates = [x.strftime('%Y-%m-%d') for x in self.date_range]
        return (f'<{cname}: date_range={dates[0]}..{dates[1]},'
                f' s_dow_corr:{has_s_dowcorr}, f_dow_corr:{has_f_dowcorr}>')


    @classmethod
    def from_doo_df(cls, df, m=18, date_range=('2020-07-01', '2099-01-01'),
                    dow=None, f_dow_corr=None, s_dow_ce=None):
        """Initialize from DataFrame.

        Parameters:

        - df: dataframe with eDOO column and multi-index.
        - date_range: (date_start, date_end) for Date_file index
        - m: number of days for estimation function
        - dow: filter by reported day of week (0=Monday, 6=Sunday), optional.
        - f_dow_corr, None, 'auto', or file-date day-of-week correction factor matreix,
          shape (7, m).  None for no correction; 'auto' for setting the correction.
        - s_dow_ce: None, 'auto, or tuple (s_dow_corr, s_dow_corr_err);
          statistics-date DoW correction factor, shape (7,), and
          the standard error (float).

        Return:

        - DOOCorrection instance.
        """

        date_range = [pd.to_datetime(x) for x in date_range]

        fdates = df.index.get_level_values('Date_file').unique().sort_values()
        fdates = fdates[(fdates >= date_range[0]) & (fdates <= date_range[1])]

        if f_dow_corr is None:
            f_dow_corr = np.ones((7, m))
        elif isinstance(f_dow_corr, str) and f_dow_corr == 'auto':
            f_dow_corr = cls.calc_fdow_correction(df, m=m, date_range=date_range)
        else:
            f_dow_corr = np.array(f_dow_corr)
            assert f_dow_corr.shape == (7, m)

        if s_dow_ce is None:
            s_dow_ce = (np.ones(7), 0.0)
        elif s_dow_ce == 'auto':
            s_dow_ce = cls.calc_sdow_correction(
                df, date_range=date_range, skip=m)
        else:
            s_dow_ce = (np.array(s_dow_ce[0]), float(s_dow_ce[1]))
            assert s_dow_ce[0].shape == (7,)

        # df1 is a working copy.
        df1 = df.loc[fdates]
        n = len(fdates)
        if n <= m:
            raise ValueError(f'm={m}: must be smaller than date range.')
        if dow is not None and n-m < 14:
            raise ValueError(f'If dow is specified, require n-m >= 14.')

        # build report matrix r[i, j], shape (n, m)
        # corresponding to eDOO at fdate=n-i, sdate=n-i-j
        rmat = np.zeros((n, m))
        dows = np.zeros(n, dtype=int) # day of week corresponding to rmat rows.
        for i in range(n):
            fdate = fdates[n-i-1]
            edoo = df1.loc[(fdate,), 'eDOO']
            rmat[i, :] = edoo[-1:-m-1:-1]
            dows[i] = fdate.dayofweek
        rmat *= f_dow_corr[dows, :]

        # If f[i+j] is the true number of cases at sdate=n-i-j,
        # then we search a function G[j] such that r[i,j] = f[i+j] * G[j].
        # We estimate f[i+j] as r[i+j-m+1, m-1]
        # Therefore G[j] = r[i, j] / r[i+j-m+1, m-1] (for all valid i)
        # with j=range(0, m)
        ii = np.arange(m, n)
        jj = np.arange(m)
        ijm1 = ii.reshape(-1, 1) + jj + (1-m)

        G = rmat[m:n, :] / rmat[ijm1, -1]
        G[~np.isfinite(G)] = 0
        # Now weighted average (more weight to high case counts).
        # also apply dow filter here.
        weights = rmat[ijm1, -1].mean(axis=1).reshape(-1, 1)

        if dow is not None:
            weights *= (dows[m:n] == dow).reshape(-1, 1)

        Gavg = np.sum(G*weights, axis=0) / weights.sum()
        sigma = (G - Gavg).std(axis=0, ddof=1)

        return cls(Gavg, sigma, [fdates[0], fdates[-1]-pd.Timedelta(m, 'd')],
                    f_dow_corr=f_dow_corr, s_dow_ce=s_dow_ce)

    @classmethod
    def calc_fdow_correction(cls, df, m=18, date_range=('2020-07-01', '2099-01-01')):
        """Return f_dow_corr matrix. Arguments as in from_doo_df().

        This is the correction based on Date_file.dayofweek
        """

        date_range = [pd.to_datetime(x) for x in date_range]
        fdates = df.index.get_level_values('Date_file').unique().sort_values()
        fdates = fdates[(fdates >= date_range[0]) & (fdates <= date_range[1])]

        if len(fdates) < 14:
            raise ValueError(f'Date range must span >= 14 days.')

        dcs = [ cls.from_doo_df(df, m=m, date_range=date_range,
               dow=dow) for dow in [0, 1, 2, 3, 4, 5, 6, None] ]

        Gs = np.array([dc.G for dc in dcs]) # (8, m) array- - values
        with np.errstate(divide='ignore'):
            fdc = Gs[7] / np.where(Gs!=0, Gs, np.nan)[:7]  # shape (7, m)

        return fdc

    @staticmethod
    def calc_sdow_correction(
            df, date_range=('2020-09-01', '2099-01-01'), skip=14, show=False,
            fname=None):
        """Calculate onset-date DoW correction factor of DOO.

        Parameters:

        - df: full summary dataframe. (will use the most recent file date.)
        - date_range: range of DOO dates to consider.
        - skip: skip this many of the most recent DOO days (because DOO statistics
          are not stable yet).
        - fname: optional filename to save plot.
        - show: whether to show a plot.


        Return:

        - dow_corr: week-day correction factor on day of onset, array (7,).
          Multiply nDOO data by this to get a smoother curve.
        - dow_corr_err: standard error on this quantity (float)
        """

        # Get data for one file date.
        fdates = df.index.get_level_values('Date_file').unique().sort_values()
        df1 = df.loc[fdates[-1]] # most recent file; new dataframe has index Date_statistics

        # Get data for range of DOO dates
        sdates = df1.index # Date_statistics
        sdmask_1 = (sdates >= date_range[0])
        sdmask_2 = (sdates <= pd.to_datetime(date_range[1]) - pd.Timedelta(skip, 'd'))
        sdmask_3 = (sdates <= fdates[-1] - pd.Timedelta(skip, 'd'))
        sdates = sdates[sdmask_1 & sdmask_2 & sdmask_3]
        ndoos = df1.loc[sdates, 'eDOO']

        # convert to log, get smooth component and deviation.
        # Savitzky-Golay filter width 15 (2 weeks) will not follow weekly oscillation.
        log_ys = np.log(ndoos)
        log_ys_sm = scipy.signal.savgol_filter(log_ys, 15, 2)
        log_ys_sm = pd.Series(log_ys_sm, index=sdates)
        log_ys_dev = log_ys - log_ys_sm

        # Statistics on per-weekday basis.
        dows = sdates.dayofweek
        dowdev = np.array([
            np.mean(log_ys_dev[dows==dow])
            for dow in range(7)
            ])
        # residual
        log_ys_dev_resid = log_ys_dev - dowdev[dows]
        # standard deviation on residual

        # Convert to per-dow correction factors and error
        dow_corr = np.exp(-dowdev)
        dow_corr_err = log_ys_dev_resid.std(ddof=1)

        if show or fname:

            fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(10, 3))
            ax = axs[0]
            ax.plot(log_ys, label='Ruwe data')
            ax.plot(ndoos.index, log_ys_sm, label='Trend', zorder=-1)
            # ax.plot(ndoos.index, log_ys_sm + dowdev[dows], label='Gecorrigeerde trend', zorder=-1)
            ax.set_ylabel('ln(nDOO)')
            ax.grid()
            ax.legend()
            ax.tick_params(axis='x', labelrotation=-20)
            ax.legend()
            # plt.xticks(rotation=-20)
            for tl in ax.get_xticklabels():
                tl.set_ha('left')

            ax = axs[1]
            ax.plot(log_ys_dev, label='t.o.v. trend')
            ax.plot(log_ys_dev_resid, label='t.o.v. gecorrigeerde trend')
            ax.set_ylabel('Verschil')
            ax.grid()
            ax.tick_params(axis='x', labelrotation=-20)
            ax.legend()
            # plt.xticks(rotation=-20)
            for tl in ax.get_xticklabels():
                tl.set_ha('left')

            # 3rd panel: per-weekday mean deviation
            ax = axs[2]
            ax.bar(np.arange(7), (np.exp(dowdev)-1)*100)
            ax.set_ylabel('Afwijking (%)')
            ax.set_xticks(np.arange(7))
            ax.set_xticklabels(['ma', 'di', 'wo', 'do', 'vr', 'za', 'zo'])

            _show_save_fig(fig, show, fname)

        return dow_corr, dow_corr_err

    def plot(self, other_Gs=None, labels=None, fname=None, show=True):
        """Plot. Optionally plot other Gdata (list) as well.

        - labels: optional list of labels for curves.
        - fname: optional plot save filename (pdf or png)
        - show: whether to show plot on-screen
        """

        fig, axs = plt.subplots(2, 1, figsize=(7, 5), tight_layout=True, sharex=True)
        # cm = ax.matshow(G)
        #fig.colorbar(cm)
        xtickloc = matplotlib.ticker.MaxNLocator(steps=[1, 2, 5])
        for ax in axs:
            ax.xaxis.set_major_locator(xtickloc)
            ax.grid()

        if other_Gs is None:
            all_Gs = [self]
        else:
            all_Gs = [self] + list(other_Gs)

        if labels is not None and len(labels) != len(all_Gs):
            raise IndexError(f'labels: mismatch in entries for number of curves.')

        if labels is None:
            labels = [
                f'{_ymd(gd.date_range[0])} .. {_ymd(gd.date_range[1])}'
                for gd in all_Gs
                ]
        ax = axs[0]

        ax.set_ylabel('DOO coverage')

        ax = axs[1]
        ax.set_ylabel('Correction factor')
        ax.set_xlabel('Days ago')

        ax.set_ylim(0.8, 5)

        color_cycle = plt.rcParams['axes.prop_cycle']()
        for gd, label in zip(all_Gs, labels):
            color = next(color_cycle)['color']
            xs = np.arange(len(gd.G))

            axs[0].plot(xs, gd.G, 'o-')
            axs[0].fill_between(
                xs, gd.G-gd.eG, gd.G+gd.eG, zorder=-1,
                color=color, alpha=0.15
                )
            axs[1].set_xlim(xs[0], xs[-1]+0.2)
            axs[1].plot(
                xs, gd.iG, 'o-',
                label=label)
            axs[1].fill_between(
                xs, gd.iG-gd.eiG, gd.iG+gd.eiG, zorder=-1,
                color=color, alpha=0.15
                )

        axs[1].legend()
        _show_save_fig(fig, show, fname)


    def create_df_nDOO(self, df):
        """Create DataFrame with nDOO, nDOO_err, ncDOO, ncDOO_err columns.

        Parameter:

        - df: DataFrame with Date_statistics index, eDOO column.
          The most recent date is assumed to be date zero.
          It won't handle missing dates.

        Return:

        - DataFrame with columns:

            - Date: date (index)
            - nDOO: estimated number of new date-of-onset cases
            - nDOO_err: estimated standard error on nDOO.
            - ncDOO, ncDOO_err: same, but corrected for DoW effects.
        """

        if df.index.name != 'Date_statistics':
            raise KeyError('df must have single index "Date_statistics".')

        df = df.sort_index()
        m = min(len(self.iG), len(df))

        # New DataFrame, initialize nDOO, nDOO_err
        new_df = pd.DataFrame(index=df.index)
        new_df['nDOO'] = df['eDOO']
        new_df['nDOO_err'] = 0.0

        dslice = slice(df.index[-m], df.index[-1])
        mslice = slice(m-1, None, -1)

        dow_corr = self.f_dow_corr[df.index[-1].dayofweek, mslice]
        new_df.loc[dslice, 'nDOO'] *=  self.iG[mslice] / dow_corr
        new_df.loc[dslice, 'nDOO_err'] = df.loc[dslice, 'eDOO'] * self.eiG[mslice] / dow_corr

        # also add ncDOO, ncDOO_err columns
        new_df['ncDOO'] = new_df['nDOO'] * self.s_dow_corr[new_df.index.dayofweek]
        ncDOO_err = new_df['nDOO'].to_numpy() * self.s_dow_corr_err
        ncDOO_err = np.sqrt(ncDOO_err**2 + new_df['nDOO_err']**2)
        new_df['ncDOO_err'] = ncDOO_err

        for col in ['nDOO', 'nDOO_err', 'ncDOO', 'ncDOO_err']:
            data = new_df.loc[dslice, col]
            data[data >= 100] = np.around(data[data >= 100])
            data = np.around(data, 1)
            new_df.loc[dslice, col] = data

        return new_df

    def plot_nDOO(self, df_eDOO, title=None, kind='nDOO', fname=None, show=True):
        """Plot one or more DataFrames with eDOO data (after corrections applied).

        Parameters:

        - df_eDOO: one dataframe or list of dataframes to plot. Each DataFrame
          must have 'Date_statistics' as index.
        - title: optional plot title string.
        - kind: 'nDOO' or 'ncDOO': the estimated new cases or the
          estimated DoW-corrected new cases; 'nncDOO' for both.
        - fname: optional filename for plot output.
        - show: whether to show on-screen.
        """

        if isinstance(df_eDOO, pd.DataFrame):
            df_eDOO = [df_eDOO]

        # which columns to plto
        if kind == 'nDOO':
            ncol, ecol = 'nDOO', 'nDOO_err'
            ylabel = 'Aantal per dag'
        elif kind == 'ncDOO':
            ncol, ecol = 'ncDOO', 'ncDOO_err'
            ylabel = 'Aantal per dag (na weekdagcorrectie)'
        else:
            raise ValueError(f'kind={kind}')

        fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5))
        color_cycle = plt.rcParams['axes.prop_cycle']()
        for dfe in df_eDOO:
            dfn = self.create_df_nDOO(dfe)

            color = next(color_cycle)['color']
            ax.semilogy(dfn[ncol], color=color, label=_ymd(dfn.index[-1]))

            dfne = dfn.iloc[-len(self.iG):]
            ax.fill_between(
                dfne.index, dfne[ncol]-dfne[ecol],
                dfne[ncol]+dfne[ecol],
                color=color, alpha=0.2, zorder=-1
                )
        ax.set_xlabel('Datum eerste ziektedag')
        ax.set_ylabel(ylabel)

        if len(df_eDOO) < 4:
            ax.legend()
        ax.grid()
        ax.grid(which='minor', axis='y')

        if title:
            ax.set_title(title)
            fig.canvas.set_window_title(title)

        _show_save_fig(fig, show, fname)


    def plot_fdow_corr(self, subtitle=None, fname=None, show=True):
        """Plot file-date DoW correction (in iG).

        Parameters:

        - subtitle: optional second line for title.
        - show: whether to show on-screen
        - fname: optional filename for plot output (png)
        """

        short_dows = 'ma,di,wo,do,vr,za,zo'.split(',')

        fig, ax = plt.subplots(tight_layout=True, figsize=(8, 4))

        igc = 1/self.f_dow_corr
        m = igc.shape[1]
        igc_pad = np.full((7, m+6), np.nan)
        for i in range(7):
            igc_pad[i, (7-i):(6-i)+m] = igc[i, 1:] # ignore noisy igc[:, 0] values
        igc_pad = (igc_pad-1)*100 # make it percentages

        vmax = min(np.nanmax(np.abs(igc_pad)), 15)

        cmap = matplotlib.cm.get_cmap('seismic')
        cmap.set_bad(color='#bbccbb')

        cm = ax.imshow(igc_pad, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
        fig.colorbar(cm)

        ax.set_xlabel('Recent ← Dag casus → Oud')
        ax.set_ylabel('Weekdag rapport')
        ax.set_yticks(np.arange(7))
        ax.set_yticklabels(short_dows)
        ax.set_xticks(np.arange(m+6))
        ax.set_xticklabels([short_dows[6-i%7] for i in range(m+6)])

        title = 'Correctiefactor nieuwe ziekmeldingen, effect (%) van rapportagedag'
        if subtitle:
            title += f'\n{subtitle}'
        ax.set_title(title)

        _show_save_fig(fig, show, fname)

def save_data_cache(df, fname):
    """Save data to pickle file in CONFIG['cache_path']."""

    fpath = CONFIG['cache_path'] / fname
    with fpath.open('wb') as f:
        pickle.dump(df, f)
    print(f'Wrote cache: {fpath}')

def load_data_cache(fname, maxageh=1):
    """Attempt to load a data from cache. Return None if failed or too old.

    maxageh is in hours.
    """

    fpath = CONFIG['cache_path'] / fname
    if not fpath.is_file():
        return None
    if fpath.stat().st_mtime < time.time() - maxageh*3600:
        print(f'Cache file {fpath} is too old.')
        return None
    with fpath.open('rb') as f:
        df = pickle.load(f)
        print(f'Loaded from cache: {fpath}')
        return df


def get_reporting_delay(df, initial_delay=7, end_trunc=4, start_trunc=5, plot=True, m=18):

    """Estimate delay from DOO (eDOO) to file_date.

    Parameters:

    - df: Dataframe with multiindex and DON, DOO, DPL, Dtot columns.
    - initial_delay: assume initial delay (days).
    - end_trunc: how many dates to truncate at the end
      (recent data unreliable; but set it too high and you'll get
       a range error. Lower than 4 is not very meaningful.)
    - start_trunc: how many dates to truncate at the beginning
      of the delay data.
    - plot: whether to show plot of the data.
    - m: days to wait for reports to converge (for weekday corrections).

    Return:

    - delays_d: pandas Series with delays (days), Date_file as index.
    """

    # Setup refdata dataframe;d
    # index: Date_statistics
    # columns: ..., nDOO: estimated number of new disease onsets.
    # Estimate is corrected for partial reporting but not for day-of-week
    # effects. Statsitics based on most recent m=18 days.
    fdates = np.array(sorted(df.index.get_level_values(0).unique()))
    fdrange = ('2020-10-01', fdates[-1])
    doo_corr = DOOCorrection.from_doo_df(
            df, date_range=fdrange, m=m,
            )
    refdata = doo_corr.create_df_nDOO(df.loc[fdates[-1]])
    refdata.loc[refdata.index[-end_trunc:],'nDOO'] = np.nan
    refdata = refdata.loc[~refdata['nDOO'].isna()]

    # df_deltas: changes in cumulative Dtot etc. values.
    # The columns 'Dtot' is roughly the 'daily case numbers'.
    columns = ['DOO', 'DON', 'DPL', 'Dtot']
    df_deltas = df[columns].groupby('Date_file').sum().diff()
    df_deltas.iloc[0, :] = 0

    # by date of disease onset
    by_doo = refdata['nDOO'].rolling(7, center=True).mean().iloc[3:-3]

    by_doo = by_doo.loc[fdates[0]-(pd.Timedelta(initial_delay-3, 'd')):]
    cum_doo = by_doo.cumsum() - by_doo[0]

    # by date of report
    by_dor = df_deltas['Dtot'].rolling(7, center=True).mean().iloc[3:-3].copy()
    cum_dor = by_dor.cumsum() - by_dor[0]

    # Get delay by matching cumulatives
    f_cumdoo2date = scipy.interpolate.interp1d(
        cum_doo, cum_doo.index.astype(int), bounds_error=False,
        fill_value=(cum_doo[0], cum_doo[-1]))

    delays = pd.Series(
        cum_dor.index - pd.to_datetime(np.around(f_cumdoo2date(cum_dor.values), -9)),
        index=cum_dor.index
        )
    delays = delays.iloc[start_trunc:]
    # delay in days
    delays_d = np.around(delays / pd.Timedelta('1 d'), 2)

    if plot:
        fig, axs = plt.subplots(3, 1, tight_layout=True, figsize=(10, 7), sharex=True)

        kwa_doo = dict(linestyle='--')

        ax = axs[0]
        ax.set_title('Aantal positieve tests (7-daags gemiddelde)')
        ax.set_ylabel('Gevallen per dag')
        ax.plot(by_dor, label='versus rapportagedatum')
        ax.plot(by_doo, label='versus 1e ziektedag', **kwa_doo)
        #    ax.set_xlim(by_doo.index[0], by_doo.index[-1])
        ax.legend()

        ax = axs[1]
        ax.set_title('Cumulatief aantal positieve tests')
        ax.set_ylabel('Cumulatieve gevallen')
        ax.plot(cum_dor, label='versus rapportagedatum')
        ax.plot(cum_doo, label='versus 1e ziektedag', **kwa_doo)
        ax.legend()

        ax = axs[2]
        ax.set_title('Tijd tussen 1e ziektedag en rapportage (versus rapportagedatum)')
        ax.plot(delays_d, label=None)
        # ax.plot(delays/pd.Timedelta('1 d') + 2.5, label='Date of infection??')
        ax.set_ylabel('Vertraging (dagen)')

        for ax in axs:
            tools.set_xaxis_dateformat(ax)

        fig.canvas.set_window_title('Rapportagevertraging')
        fig.show()

    return delays_d


