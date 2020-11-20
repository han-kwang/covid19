#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for casus analysis. Import this.

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

Created on Sun Nov  8 22:44:09 2020

@author: @hk_nien
"""

import re
import locale
import pickle
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import urllib
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Note: need to run this twice before NL locale takes effect.
locale.setlocale(locale.LC_ALL, 'nl_NL.UTF-8')


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
    )

def set_conf(**kwargs):
    """Set configuration parameters (CONFIG global variable)."""

    for k, v in kwargs:
        if k in CONFIG:
            CONFIG[k] = type(CONFIG(k))(v)
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


def load_casus_data(date):
    """Return DataFrame with casus data from one RIVM csv file.

    Parameter:

    - date: 'yyyy-mm-dd' string

    Load from data-casus/COVID-19_casus_landelijk_xxx.csv.gz
    Date_file and Date_statistics will be timestamps (at time 0:00)
    """

    fname = f'{CONFIG["cdpath"]}/COVID-19_casus_landelijk_{date}.csv'
    if Path(fname).exists():
       pass
    elif Path(f'{fname}.gz').exists():
        fname = f'{fname}.gz'
    else:
        raise FileNotFoundError(f'{fname} or {fname}.gz')

    df = pd.read_csv(fname, sep=',')

    # Convert 'yyyy-mm-dd 10:00' to a 'yyyy-mm-dd' timestamp, because the 10:00
    # is just clutter when comparing dates.
    ymd = df['Date_file'].str.extract(r'(....-..-..)', expand=False)
    df['Date_file'] = pd.to_datetime(ymd)
    df['Date_statistics'] = pd.to_datetime(df['Date_statistics'])
    return df

def load_casus_summary(date):
    """Return summary dataframe for this date.

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

def download_rivm_casus_files():
    """Download missing day files in data-casus.

    Download from here:
    https://github.com/mzelst/covid-19/tree/master/data-rivm/casus-datasets

    The CSV layout is slightsly different from the RIVM data.
    The files as downloaded are huge, though highly compressible and stored
    here in gzipped format. To download all files rather than a few
    incrementally, it may be better to download a repository ZIP file from
    Github.

    files /COVID-19_casus_landelijk_yyyy-mm-dd.csv.gz.

    Return: number of files downloaded.
    """

    fdates = set()
    for fn in CONFIG["cdpath"].glob('COVID-19_casus_landelijk_*.csv.gz'):
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
    if tm_now.hour < 15:
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
              'huge data files? Ctrl-C to abort, ENTER to continue.')

    fname_template = 'COVID-19_casus_landelijk_{date}.csv'
    url_template = (
        'https://github.com/mzelst/covid-19/raw/master/data-rivm/'
        f'casus-datasets/{fname_template}')

    for fdate in fdates_missing:
        url = url_template.format(date=fdate)
        # @@ debug
        # url = 'https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_cumulatief.csvx'
        print(f'Downloading casus data for {fdate} ...')
        fpath = CONFIG["cdpath"] / (fname_template.format(date=fdate) + '.gz')
        with urllib.request.urlopen(url) as response:
            data_bytes = response.read()
            if response.code != 200:
                raise FileNotFoundError(f'Response {response.code} on {url}')
            if ('Date_statistics' not in data_bytes[:100].decode('utf-8')
                or fdate not in data_bytes[-200:].decode('utf-8')):
                # RIVM website for one does not give a 404 when it should
                raise ValueError(f'Bad or incomplete data in {url}.')

        with gzip.open(fpath, 'wb') as f:
            f.write(data_bytes)
        print(f'Wrote {fpath} .')
        return len(fdates_missing)


def _load_one_df(date):
    """Helper function for load_merged_summary in multiproc pool."""

    # print(f'({date}) ', end='', flush=True)
    print('.', end='', flush=True)
    return load_casus_summary(date).reset_index()

def load_merged_summary(date_lo, date_hi):
    """Return merged summary DataFrame between two yyyy-mm-dd file dates.

    It loads from the csv file as created by create_merged_summary_csv().
    """

    # Build list of file dates to read
    date_lo = pd.to_datetime(date_lo)
    date_hi = pd.to_datetime(date_hi)
    date = date_lo
    dfsums = []
    fdates = []
    while date <= date_hi:
        date_str = date.strftime('%Y-%m-%d')
        fpath = CONFIG["cdpath"] / f'COVID-19_casus_landelijk_{date_str}.csv.gz'
        if not fpath.is_file():
            print(f'Using casus data before {date_str}.')
            break
        fdates.append(date_str)
        date += pd.Timedelta(1, 'd')

    msg = f'Loading casus data for {len(fdates)} days (using {{ncpu}} processes)'
    with PoolNCPU(msg) as pool:
        dfsums = pool.map(_load_one_df, fdates)
        print()
    dfsmerged = pd.concat(dfsums)
    return dfsmerged

def create_merged_summary_csv(date_lo='2020-07-01'):
    """Load lots fo data, write to data/casus_history_summary.csv."""

    dfmerged = load_merged_summary(date_lo, '2099-01-01')
    fpath = CONFIG["dpath"] / 'casus_history_summary.csv'
    dfmerged.to_csv(fpath)
    print(f'Wrote {fpath} .')

def load_merged_summary_csv(date_lo='2020-07-01', date_hi='2099-01-01'):
    """Return history summary dataframe.

    Columns: Date_file, Date_statistics, DOO, DON, DPL, Dtot.
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

    for dtf in dtfs:
        print(f'Processing {dtf.strftime("%Y-%m-%d")}.')
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

    return df


def _ymd(date):
    return date.strftime('%Y-%m-%d')


class DOOCorrection:
    """Coverage of recent DOO reports.

    For a stat_date that is j days before the file_date, G[j]
    is the estimated fraction of the true number of DOO cases for
    that stat_date.

    The calculation method is essentially the method
    proposed by Twitter @bslagter .

    Attributes:

    - G: array (m,) with G values
    - eG: array (m,) with estimated errors in G
    - date_range: [date_lo, date_hi]
    - G_dow_corr: DoW correction factors for G, shape (7, m).
      (use G*G_dow_corr[dow] or iG/G_dow_corr[dow])

    - iG: inverse G
    - eiG: standard error on iG

    Functions:

    - __init__(): direct initialization of all attributes
    - from_doo_df(): analyze casus dataframe
    - plot(): plot
    """

    def __init__(self, G, eG, date_range, G_dow_corr):
        """Init attributes directly."""

        self.G = np.array(G)
        self.eG = np.array(eG)
        self.date_range = [pd.to_datetime(x) for x in date_range[:2]]

        self.iG = 1/np.where(self.G!=0, self.G, np.nan)
        mask_big_err = (self.G < 2*self.eG)
        mask_big_iG = (np.where(np.isfinite(self.iG), self.iG, 999)  > 5)

        self.iG[mask_big_err | mask_big_iG] = np.nan
        self.eiG = self.eG*self.iG**2

        self.G_dow_corr = np.array(G_dow_corr)


    @classmethod
    def from_doo_df(cls, df, m=18, date_range=('2020-07-01', '2099-01-01'),
                    dow=None, G_dow_corr=None):
        """Initialize from DataFrame.

        Parameters:

        - df: dataframe with eDOO column and multi-index.
        - date_range: (date_start, date_end) for Date_file index
        - m: number of days for estimation function
        - dow: filter by reported day of week (0=Monday, 6=Sunday), optional.
        - G_dow_corr, optional day-of-week correction factor matrix,
          shape (7, m). Default: ones.

        Return:

        - G: array, shape (m,)
        - sigma_G: estimated standard error based on the input data.
        """

        date_range = [pd.to_datetime(x) for x in date_range]

        fdates = df.index.get_level_values('Date_file').unique().sort_values()
        fdates = fdates[(fdates >= date_range[0]) & (fdates <= date_range[1])]

        if G_dow_corr is None:
            G_dow_corr = np.ones((7, m))
        else:
            assert G_dow_corr.shape == (7, m)

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
        rmat *= G_dow_corr[dows, :]

        # If f[i+j] is the true number of cases at sdate=n-i-j,
        # then we search a function G[j] such that r[i,j] = f[i+j] * G[j].
        # We estimate f[i+j] as r[i+j-m, m]
        # Therefore G[j] = r[i, j] / r[i+j-m+1, m] (for all valid i)
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

        return cls(Gavg, sigma, [fdates[0], fdates[-1]], G_dow_corr)

    @classmethod
    def calc_dow_correction(cls, df, m=18, date_range=('2020-07-01', '2099-01-01')):
        """Return G_dow_corr matrix. Arguments as in from_doo_df()."""

        date_range = [pd.to_datetime(x) for x in date_range]
        fdates = df.index.get_level_values('Date_file').unique().sort_values()
        fdates = fdates[(fdates >= date_range[0]) & (fdates <= date_range[1])]

        if len(fdates) < 14:
            raise ValueError(f'Date range must span >= 14 days.')

        gds = [
               cls.from_doo_df(df, m=18, date_range=date_range, dow=dow)
               for dow in [0, 1, 2, 3, 4, 5, 6, None]
            ]

        Gs = np.array([gd.G for gd in gds]) # (8, m) array
        with np.errstate(divide='ignore'):
            gdc = Gs[7] / np.where(Gs!=0, Gs, np.nan)[:7]  # shape (7, m)

        return gdc


    def plot(self, other_Gs=None, labels=None):
        """Plot. Optionally plot other Gdata (list) as well.

        - labels: optional list of labels for curves.
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
        fig.show()

    def create_df_nDOO(self, df):
        """Create DataFrame with nDOO, nDOO_err columns.

        Parameter:

        - df: DataFrame with Date_statistics index, eDOO column.
          The most recent date is assumed to be date zero.
          It won't handle missing dates.

        Return:

        - DataFrame with columns:

            - Date: date (index)
            - nDOO: estimated number of new date-of-onset cases
            - nDOO_err: estimated standard error on nDOO.
        """

        if df.index.name != 'Date_statistics':
            raise KeyError('df must have single index "Date_statistics".')

        df = df.sort_index()
        m = min(len(self.iG), len(df))

        new_df = pd.DataFrame(index=df.index)
        new_df['nDOO'] = df['eDOO']
        new_df['nDOO_err'] = 0.0

        dslice = slice(df.index[-m], df.index[-1])
        mslice = slice(m-1, None, -1)

        dow_corr = self.G_dow_corr[df.index[-1].dayofweek, mslice]
        new_df.loc[dslice, 'nDOO'] *=  self.iG[mslice] / dow_corr
        new_df.loc[dslice, 'nDOO_err'] = df.loc[dslice, 'eDOO'] * self.eiG[mslice] / dow_corr

        for col in ['nDOO', 'nDOO_err']:
            data = new_df.loc[dslice, col]
            data[data >= 100] = np.around(data[data >= 100], 0)
            data = np.around(data, 1)
            new_df.loc[dslice, col] = data

        return new_df

    def plot_nDOO(self, df_eDOO, title=None):
        """Plot one or more DataFrames with eDOO data (after corrections applied).

        Parameters:

        - df_eDOO: one dataframe or list of dataframes to plot. Each DataFrame
          must have 'Date_statistics' as index.
        - title: optional plot title string.
        """

        if isinstance(df_eDOO, pd.DataFrame):
            df_eDOO = [df_eDOO]

        fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5))
        color_cycle = plt.rcParams['axes.prop_cycle']()
        for dfe in df_eDOO:
            dfn = self.create_df_nDOO(dfe)

            color = next(color_cycle)['color']
            ax.semilogy(dfn['nDOO'], color=color, label=_ymd(dfn.index[-1]))

            dfne = dfn.iloc[-len(self.iG):]
            ax.fill_between(
                dfne.index, dfne['nDOO']-dfne['nDOO_err'],
                dfne['nDOO']+dfne['nDOO_err'],
                color=color, alpha=0.2, zorder=-1
                )
        ax.set_xlabel('Datum eerste ziektedag')
        ax.set_ylabel('Aantal')

        if len(df_eDOO) < 4:
            ax.legend()
        ax.grid()

        if title:
            ax.set_title(title)
            fig.canvas.set_window_title(title)

        fig.show()

    def plot_dow_corr(self, subtitle=None):
        """Plot DoW correction (in iG).

        Parameters:

        - subtitle: optional second line for title.
        """

        short_dows = 'ma,di,wo,do,vr,za,zo'.split(',')

        fig, ax = plt.subplots(tight_layout=True, figsize=(8, 4))

        igc = 1/self.G_dow_corr
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
        fig.show()



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
