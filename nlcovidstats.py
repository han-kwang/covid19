#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:51:39 2020

This is best run inside Spyder, not as standalone script.

Author: @hk_nien on Twitter.
"""
import re
import sys
import io
import urllib
import urllib.request
from pathlib import Path
import time
import locale
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import nl_regions
import scipy.signal
import scipy.interpolate
import scipy.integrate
import tools
from g_mobility_data import get_g_mobility_data
from nlcovidstats_data import (
    init_data,
    DFS,
    get_municipalities_by_pop,
    load_cumulative_cases,
    )


# These delay values are tuned to match the RIVM Rt estimates.
# The represent the delay (days) from infection to report date,
# referencing the report date.
# Extrapolation: constant value.
DELAY_INF2REP = [
    ('2020-07-01', 7.5),
    ('2020-09-01', 7),
    ('2020-09-15', 9),
    ('2020-10-09', 9),
    ('2020-11-08', 7),
    ('2020-12-01', 6.5),
    ('2021-02-15', 6.5),
    ('2021-04-05', 4),
    ('2021-07-06', 4),
    ('2021-07-15', 5),
    ('2021-07-23', 4),
    ('2021-07-30', 4),
    ('2021-11-04', 4),
    ('2021-11-11', 4.5),
    ('2021-11-20', 5),
    ('2021-11-25', 5),
    ('2021-12-04', 4.5), # test capacity increased
    ('2021-12-08', 4),
    ('2022-01-04', 4),
    ('2022-01-08', 4.3), # speculation
    ('2022-01-11', 4.3), # speculation
    ]




_DOW_CORR_CACHE = {} # keys: dayrange tuples.

def get_dow_correction_rolling(nweeks=7, taper=0.5):
    """Return DoW correction factors for all dates.

    Parameters:

    - nweeks: number of preceding weeks to use for each date.
    - taper: which fraction of old data to taper to lower weight.

    Return:

    - Series with same timestamp index as cases data.
    """
    df, _ = get_region_data('Nederland', lastday=-1, correct_dow=None)
    # df = df.iloc[3:-3].copy() # strip edge points without well defined 7d mean.
    # Correction factor - 1
    df['Delta_factor'] = df['Delta']/df['Delta7r']
    ntaper = int(nweeks*taper + 0.5)
    kernel = np.zeros(nweeks*2 + 1)
    kernel[-nweeks:] = 1
    kernel[-nweeks:-nweeks+ntaper] = np.linspace(1/ntaper, 1-1/ntaper, ntaper)
    kernel /= kernel.sum()
    df['Dow_factor'] = np.nan
    for idow in range(7):
        row_select = df.index[df.index.dayofweek == idow]
        facs = df.loc[row_select, 'Delta_factor']
        n = len(facs)
        assert len(facs) > nweeks
        mean_factors = np.convolve(facs, kernel, mode='same')
        mean_factors[mean_factors == 0] = np.nan
        df.loc[row_select, 'Dow_factor'] = 1/mean_factors
    df.loc[df.index[:8], 'Dow_factor'] = np.nan
    return df['Dow_factor']



def get_dow_correction(dayrange=(-50, -1), verbose=False):
    """Return array with day-of-week correction factors.

    - dayrange: days to consider for DoW correction.
    - verbose: whether to show plots and print diagnostics.

    Return:

    - dow_corr_factor: array (7,) with DoW correction (0=Monday).
    """

    dayrange = tuple(dayrange)
    if dayrange in _DOW_CORR_CACHE and not verbose:
        return _DOW_CORR_CACHE[dayrange].copy()

    # timestamp index, columns Delta, Delta7r, and others.
    df, _ = get_region_data('Nederland', lastday=dayrange[-1], correct_dow=None)
    df = df.iloc[:-4] # Discard the last rows that have no correct rolling average.
    df = df.iloc[dayrange[0]-dayrange[1]:]

    # Correction factor - 1
    df['Delta_factor'] = df['Delta']/df['Delta7r']

    # Collect by day of week (0=Monday)
    factor_by_dow = np.zeros(7)
    for i in range(7):
        factor_by_dow[i] = 1 / df.loc[df.index.dayofweek == i, 'Delta_factor'].mean()
    factor_by_dow /= factor_by_dow.mean()


    df['Delta_est_factor'] = factor_by_dow[df.index.dayofweek]
    df['Delta_corrected'] = df['Delta'] * df['Delta_est_factor']

    rms_dc = (df['Delta_corrected']/df['Delta7r']).std()
    rms_d = df['Delta_factor'].std()

    if verbose:
        print('DoW effect: deviations from 7-day rolling average.\n'
              f'  Original: RMS={rms_d:.3g}; after correction: RMS={rms_dc:.3g}')

        fig, ax = plt.subplots(tight_layout=True)

        ax.plot(df['Delta_factor'], label='Delta')
        ax.plot(df['Delta_corrected'] / df['Delta7r'], label='Delta_corrected')
        ax.plot(df['Delta_est_factor'], label='Correction factor')

        tools.set_xaxis_dateformat(ax, 'Date')
        ax.legend()
        ax.set_ylabel('Daily cases deviation')

        title = 'Day-of-week correction on daily cases'
        ax.set_title(title)
        fig.canvas.manager.set_window_title(title)
        fig.show()

    if rms_dc > 0.8*rms_d:
        print(f'WARNING: DoW correction for dayrange={dayrange} does not seem to work.\n'
              '  Abandoning this correction.')

        factor_by_dow = np.ones(7)

    _DOW_CORR_CACHE[dayrange] = factor_by_dow.copy()
    return factor_by_dow


def get_rolling7_with_est3(y, pattern_weeks, m=3):
    """Create rolling 7-d average with last three days estimated.

    Parameters:

    - y: 1D array or Series with values (1 per day).
    - pattern_weeks: how many previous weeks to estimate weekday patterns.
    - m: how many days to estimate initial derivative.

    Return:

    - y7: centered rolling-7 average, last 3 days estimated, as array.
    """
    # Rolling average
    ker = np.full(7, 1/7)
    y = np.array(y)
    n, = y.shape
    y7 = np.convolve(y, ker, mode='same')
    y7[:3] = y7[-3:] = np.nan

    # DoW correction on last 3 days
    n = len(y)
    for i0 in range(n-3, n):
        ii = np.arange(i0-7*pattern_weeks, i0, 7)
        ratio = (y7[ii]/y[ii]).mean()
        y7[i0] = y[i0]*ratio

    # get slope a1 of last days
    assert m > 0
    a0 = y7[-4]
    a1 = (y7[-4] - y7[-4-m]) / m

    # last four points are y0 .. y3 for x=0, .., 3
    # fit a polynomial a0 + a1*x + a2*x**2.
    yy = y7[-3:]
    xx = np.arange(1, 4)
    a2, _, _, _  = np.linalg.lstsq(
        (xx**2)[:, np.newaxis],
        yy - a0 - a1*xx,
        rcond=None
        )

    y7[-3:] = a0 + a1*xx + a2*xx**2
    y7[:3] = y7[3] # just pad on the left

    return y7


def get_region_data(region, lastday=-1, printrows=0, correct_anomalies=True,
                    correct_dow='r7'):
    """Get case counts and population for one municipality.

    It uses the global DFS['mun'], DFS['cases'] dataframe.

    Parameters:

    - region: region name (see below)
    - lastday: last day to include.
    - printrows: print this many of the most recent rows
    - correct_anomalies: correct known anomalies (hiccups in reporting)
      by reassigning cases to earlier dates.
    - correct_dow: ignored, for backward compatibility.

    Special municipalities:

    - 'Nederland': all
    - 'HR:Zuid', 'HR:Noord', 'HR:Midden', 'HR:Midden+Zuid', 'HR:Midden+Noord':
      holiday regions.
    - 'MS:xx-yy': municipalities with population xx <= pop/1000 < yy'
    - 'P:xx': province

    Use data up to lastday.

    Return:

    - df: dataframe with added columns:

        - Delta: daily increase in case count (per capita).
        - Delta_dowc: daily increase, day-of-week correction applied
          based on national pattern in most recent 7 weeks.
        - Delta7r: daily increase as 7-day rolling average
          (last 3 days are estimated).
        - DeltaSG: daily increase, smoothed with (15, 2) Savitsky-Golay filter.Region selec
    - pop: population.
    """

    df1, npop = nl_regions.select_cases_region(DFS['cases'], region)

    # df1 will have index 'Date_of_report', columns:
    # 'Total_reported', 'Hospital_admission', 'Deceased'

    assert correct_dow in [None, 'r7']
    if lastday < -1 or lastday > 0:
        df1 = df1.iloc[:lastday+1]

    if len(df1) == 0:
        raise ValueError(f'No data for region={region!r}.')

    # nc: number of cases
    nc = df1['Total_reported'].diff()
    if printrows > 0:
        print(nc[-printrows:])

    nc.iat[0] = 0
    df1['Delta'] = nc/npop
    if correct_anomalies:
        _correct_delta_anomalies(df1)
        nc = df1['Delta'] * npop
    nc7 = get_rolling7_with_est3(nc, 3, 3)
    df1['Delta7r'] = nc7/npop
    df1['DeltaSG'] = scipy.signal.savgol_filter(
        nc/npop, 15, 2, mode='interp')

    return df1, npop

def _correct_delta_anomalies(df):
    """Apply anomaly correction to 'Delta' column.

    Store original values to 'Delta_orig' column.
    Pull data from DFS['anomalies']
    """

    dfa = DFS['anomalies']
    df['Delta_orig'] = df['Delta'].copy()

    dt_tol = pd.Timedelta(12, 'h') # tolerance on date matching
    match_date = lambda dt: abs(df.index - dt) < dt_tol
    preserve_n = True

    for (date, data) in dfa.iterrows():
        if date == '2021-02-08':
            print('@foo')
        f = data['fraction']
        dt = data['days_back']
        dn = df.loc[match_date(date), 'Delta_orig'] * f
        if len(dn) == 0:
            print(f'Anomaly correction: no match for {date}; skipping.')
            continue
        assert len(dn) == 1
        dn = dn[0]

        df.loc[match_date(date + pd.Timedelta(dt, 'd')), 'Delta'] += dn
        if dt != 0:
            df.loc[match_date(date), 'Delta'] -= dn
        else:
            preserve_n = False

    if preserve_n:
        assert np.isclose(df["Delta"].sum(), df["Delta_orig"].sum(), rtol=1e-6, atol=0)
    else:
        delta = df["Delta"].sum() - df["Delta_orig"].sum()
        print(f'Note: case count increased by {delta*17.4e6:.0f} cases due to anomalies.')



def construct_Dfunc(delays, plot=False):
    """Return interpolation functions fD(t) and fdD(t).

    fD(t) is the delay between infection and reporting at reporting time t.
    fdD(t) is its derivative.

    Parameter:

    - delays: tuples (datetime_report, delay_days). Extrapolation is at
      constant value.
    - plot: whether to generate a plot.

    Return:

    - fD: interpolation function for D(t) with t in nanoseconds since epoch.
    - fdD: interpolation function for dD/dt.
      (taking time in ns but returning dD per day.)
    - delay_str: delay string e.g. '7' or '7-9'
    """

    ts0 = [float(pd.to_datetime(x[0]).to_datetime64()) for x in delays]
    Ds0 = [float(x[1]) for x in delays]
    if len(delays) == 1:
        # prevent interp1d complaining.
        ts0 = [ts0[0], ts0[0]+1e9]
        Ds0 = np.concatenate([Ds0, Ds0])

    # delay function as linear interpolation;
    # nanosecond timestamps as t value.
    fD0 = scipy.interpolate.interp1d(
        ts0, Ds0, kind='linear', bounds_error=False,
        fill_value=(Ds0[0], Ds0[-1])
    )

    # construct derivative dD/dt, smoothen out
    day = 1e9*86400 # one day in nanoseconds
    ts = np.arange(ts0[0]-3*day, ts0[-1]+3.01*day, day)
    dDs = (fD0(ts+3*day) - fD0(ts-3*day))/6
    fdD = scipy.interpolate.interp1d(
        ts, dDs, 'linear', bounds_error=False,
        fill_value=(dDs[0], dDs[-1]))

    # reconstruct D(t) to be consistent with the smoothened derivative.
    Ds = scipy.integrate.cumtrapz(dDs, ts/day, initial=0) + Ds0[0]
    fD = scipy.interpolate.interp1d(
        ts, Ds, 'linear', bounds_error=False,
        fill_value=(Ds[0], Ds[-1]))

    Dmin, Dmax = np.min(Ds0), np.max(Ds0)
    if Dmin == Dmax:
        delay_str = f'{Dmin:.0f}'
    else:
        delay_str = f'{Dmin:.0f}-{Dmax:.0f}'

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3), tight_layout=True)
        tsx = np.linspace(
            ts[0],
            int(pd.to_datetime('now').to_datetime64())
            )
        ax.plot(pd.to_datetime(tsx.astype(np.int64)), fD(tsx))
        ax.set_ylabel('Vertraging (dagen)')
        tools.set_xaxis_dateformat(ax, 'Rapportagedatum')
        title = 'Vertraging = t_rapportage - t_infectie - t_generatie/2'
        fig.canvas.manager.set_window_title(title)
        ax.set_title(title)
        fig.show()

    return fD, fdD, delay_str


def estimate_Rt_df(r, delay='DEFAULT', Tc=4.0):
    """Return Rt data, assuming delay infection-reporting.

    - r: Series with smoothed new reported cases.
      (e.g. 7-day rolling average or other smoothed data).
    - delay: assume delay days from infection to positive report.
      alternatively: list of (timestamp, delay) tuples if the delay varies over time.
      'DEFAULT' for default list.
      The timestamps refer to the date of report.
    - Tc: assume generation interval.

    Return:

    - DataFrame with columns 'Rt' and 'delay'.
    """
    if isinstance(delay, str) and delay == 'DEFAULT':
        delay = DELAY_INF2REP
    if not hasattr(delay, '__getitem__'):
        # simple delay - attach data to index with proper offset
        log_r = np.log(r.to_numpy()) # shape (n,)
        assert len(log_r.shape) == 1
        log_slope = (log_r[2:] - log_r[:-2])/2 # (n-2,)
        Rt = np.exp(Tc*log_slope) # (n-2,)

        index = r.index[1:-1] - pd.Timedelta(delay, unit='days')
        Rdf = pd.DataFrame(
            dict(Rt=pd.Series(index=index, data=Rt, name='Rt'))
            )
        Rdf['delay'] = delay
    else:
        # the hard case: delay varies over time.
        # if ri is the rate of infections, tr the reporting date, and D
        # the delay, then:
        # ri(tr-D(tr)) = r(tr) / (1 - dD/dt)
        fD, fdD, _ = construct_Dfunc(delay)

        # note: timestamps in nanoseconds since epoch, rates in 'per day' units.
        day_ns = 86400e9
        tr = r.index.view(np.int64)
        ti = tr - fD(tr) * day_ns
        ri = r.to_numpy() / (1 - fdD(tr))

        # now get log-derivative the same way as above
        log_ri = np.log(np.where(ri==0, np.nan, ri))
        log_slope = (log_ri[2:] - log_ri[:-2])/2 # (n-2,)
        Rt = np.exp(Tc*log_slope) # (n-2,)

        # build series with timestamp index
        # (Note: int64 must be specified explicitly in Windows, 'int' will be
        # int32.)
        Rt_series = pd.Series(
            data=Rt, name='Rt',
            index=pd.to_datetime(ti[1:-1].astype(np.int64))
        )
        Rdf = pd.DataFrame(dict(Rt=Rt_series))
        Rdf['delay'] = fD(tr[1:-1])

    return Rdf





def get_t2_Rt(ncs, delta_t, i0=-3):
    """Return most recent doubling time and Rt, from case series"""

    # exponential fit
    t_gen = 4.0 # generation time (d)
    t_double = delta_t / np.log2(ncs.iloc[i0]/ncs.iloc[i0-delta_t])
    Rt = 2**(t_gen / t_double)
    return t_double, Rt

def add_labels(ax, labels, xpos, mindist_scale=1.0, logscale=True):
    """Add labels, try to have them avoid bumping.


    - labels: list of tuples (y, txt)
    - mindist_scale: set to >1 or <1 to tweak label spacing.
    """
    from scipy.optimize import fmin_cobyla

    ymin, ymax = ax.get_ylim()

    if logscale:
        mindist = np.log10(ymax/ymin)*0.025*mindist_scale
    else:
        mindist = (ymax - ymin)*0.025*mindist_scale


    labels = sorted(labels)

    # log positions and sorted$ffmpeg -i Rt_%03d.png  -c:v libx264 -r 25 -pix_fmt yuv420p out.mp4

    if logscale:
        Ys = np.log10([l[0] for l in labels])
    else:
        Ys = np.array([l[0] for l in labels])
    n = len(Ys)

    # Distance matrix: D @ y = distances between adjacent y values
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i, i] = -1
        D[i, i+1] = 1

    def cons(Y):
        ds = D @ Y
        errs = np.array([ds - mindist, ds])
        #print(f'{np.around(errs, 2)}')
        return errs.reshape(-1)

    # optimization function
    def func(Y):
        return ((Y - Ys)**2).sum()

    new_Ys = fmin_cobyla(func, Ys, cons, catol=mindist*0.05)

    for Y, (_, txt) in zip(new_Ys, labels):
        y = 10**Y if logscale else Y
        ax.text(xpos, y, txt, verticalalignment='center')


def _zero2nan(s):
    """Return copy of array/series s, negative/zeros replaced by NaN."""

    sc = s.copy()
    sc[s <= 0] = np.nan
    return sc

def _add_event_labels(ax, tmin, tmax, with_ribbons=True, textbox=False, bottom=True,
                            flagmatch='RGraph'):
    """Add event labels and ribbons to axis (with date on x-axis).

    - ax: axis object
    - tmin, tmax: time range to assume for x axis.
    - textbox: whether to draw text in a semi-transparent box.
    - bottom: whether to put labels at the bottom rather than top.
    - flagmatch: which flags to match (regexp).
    """

    ymin, ymax = ax.get_ylim()
    y_lab = ymin if bottom else ymax
    ribbon_yspan =  (ymax - ymin)*0.35
    ribbon_hgt = ribbon_yspan*0.1 # ribbon height
    ribbon_ystep = ribbon_yspan*0.2
    df_events = DFS['events']
    ribbon_colors = ['#ff0000', '#cc7700'] * 10
    if df_events is not None:
        i_res = 0
        for _, (res_t, res_t_end, res_d, flags) in df_events.reset_index().iterrows():
            if not (tmin <= res_t <= tmax):
                continue
            if flags and not re.match(flagmatch, flags):
                continue
            res_d = res_d.replace('\\n', '\n')
            # note; with \n in text, alignment gets problematic.
            txt = ax.text(res_t, y_lab, f'  {res_d}', rotation=90, horizontalalignment='center',
                          verticalalignment='bottom' if bottom else 'top',
                          fontsize=8)
            if textbox:
                txt.set_bbox(dict(facecolor='white', alpha=0.4, linewidth=0))
            if pd.isna(res_t_end):
                continue
            if with_ribbons:
                res_t_end = min(res_t_end, tmax)
                a, b = (ribbon_ystep * i_res), (ribbon_yspan - ribbon_hgt)
                rect_y_lo = a % b + y_lab
                color = ribbon_colors[int(a // b)]

                rect = matplotlib.patches.Rectangle((res_t, rect_y_lo), res_t_end-res_t, ribbon_hgt,
                                                    color=color, alpha=0.15, lw=0, zorder=20)
                ax.add_patch(rect)
            i_res += 1


def plot_daily_trends(ndays=100, lastday=-1, mun_regexp=None, region_list=None,
                      source='r7', subtitle=None):
    """Plot daily-case trends (pull data from global DFS dict).

    - lastday: up to this day.
    - source: 'r7' (7-day rolling average), 'raw' (no smoothing), 'sg'
      (Savitsky-Golay smoothed).
    - mun_regexp: regular expression matching municipalities.
    - region_list: list of municipalities (including e.g. 'HR:Zuid',
      'POP:100-200', 'JSON:{...}'.
      if mun_regexp and mun_list are both specified, then concatenate.
      If neither are specified, assume 'Nederland'.

      JSON is a json-encoded dict with:

      - 'label': short label string
      - 'color': for plotting, optional.
      - 'fmt': format for plotting, e.g. 'o--', optional.
      - 'muns': list of municipality names

    - subtitle: second title line (optional)
    """

    df_events = DFS['events']
    df_mun = DFS['mun']

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(top=0.945-0.03*(subtitle is not None),
                        bottom=0.1, left=0.09, right=0.7)


    if region_list is None:
        region_list = []

    if mun_regexp:
        region_list = [m for m in df_mun.index if re.match(mun_regexp, m)] + region_list

    if region_list == []:
        region_list = ['Nederland']

    labels = [] # tuples (y, txt)f
    citystats = [] # tuples (Rt, T2, cp100k, cwk, popk, city_name)
    for region in region_list:
        df1, n_inw = get_region_data(region, lastday=lastday)
        df1 = df1.iloc[-ndays:]
        fmt = 'o-' if ndays < 70 else '-'
        psize = 5 if ndays < 30 else 3

        dnc_column = dict(r7='Delta7r', raw='Delta', sg='DeltaSG')[source]

        if region.startswith('JSON:'):
            reg_dict = json.loads(region[5:])
            reg_label = reg_dict['label']
            if 'fmt' in reg_dict:
                fmt = reg_dict['fmt']
            color = reg_dict['color'] if 'color' in reg_dict else None
        else:
            reg_label = re.sub(r'POP:(.*)-(.*)', r'\1k-\2k inw.', region)
            reg_label = re.sub(r'^[A-Z]+:', '', reg_label)
            color = None

        ax.semilogy(df1[dnc_column]*1e5, fmt, color=color, label=reg_label, markersize=psize)
        delta_t = 7
        i0 = dict(raw=-1, r7=-3, sg=-3)[source]
        t_double, Rt = get_t2_Rt(df1[dnc_column], delta_t, i0=i0)
        citystats.append((np.around(Rt, 2), np.around(t_double, 2),
                          np.around(df1['Delta'][-1]*1e5, 2),
                          int(df1['Delta7r'][-4] * n_inw * 7 + 0.5),
                          int(n_inw/1e3 + .5), reg_label))

        if abs(t_double) > 60:
            texp = f'Stabiel'
        elif t_double > 0:
            texp = f'×2: {t_double:.3g} d'
        elif t_double < 0:
            texp = f'×½: {-t_double:.2g} d'

        ax.semilogy(
            df1.index[[i0-delta_t, i0]], df1[dnc_column].iloc[[i0-delta_t, i0]]*1e5,
            'k--', zorder=-10)

        labels.append((df1[dnc_column][-1]*1e5, f' {reg_label} ({texp})'))

    _add_event_labels(
        ax, df1.index[0], df1.index[-1], with_ribbons=False,
        flagmatch='CaseGraph'
        )


    dfc = pd.DataFrame.from_records(
        sorted(citystats), columns=['Rt', 'T2', 'C/100k', 'C/wk', 'Pop/k', 'Region'])
    dfc.set_index('Region', inplace=True)
    print(dfc)


    lab_x = df1.index[-1] + pd.Timedelta('1.2 d')
    add_labels(ax, labels, lab_x)

    if source == 'r7':
        ax.axvline(df1.index[-4], color='gray')
        # ax.text(df1.index[-4], 0.3, '3 dagen geleden - extrapolatie', rotation=90)
        title = '7-daags voortschrijdend gemiddelde; laatste 3 dagen zijn een schatting'
    elif source == 'sg':
        ax.axvline(df1.index[-8], color='gray')
        # ax.text(df1.index[-4], 0.3, '3 dagen geleden - extrapolatie', rotation=90)
        title = 'Gefilterde data; laatste 7 dagen zijn minder nauwkeurig'
    else:
        title = 'Dagcijfers'

    ax.set_ylabel('Nieuwe gevallen per 100k per dag')

    #ax.set_ylim(0.05, None)
    ax.set_xlim(None, df1.index[-1] + pd.Timedelta('1 d'))
    from matplotlib.ticker import LogFormatter, FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    # Monkey-patch to prevent '%e' formatting.
    LogFormatter._num_to_string = lambda _0, x, _1, _2: ('%g' % x)
    ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(3, 1)))
    #plt.xticks(pd.to_dateTime(['2020-0{i}-01' for i in range(1, 9)]))
    ax.legend() # loc='lower left')

    tools.set_xaxis_dateformat(ax, yminor=True)

    if subtitle:
        title += f'\n{subtitle}'
        win_xtitle = f', {subtitle}'
    else:
        win_xtitle = ''

    ax.set_title(title)
    fig.canvas.manager.set_window_title(f'Case trends (ndays={ndays}){win_xtitle}')
    fig.show()


def plot_cumulative_trends(ndays=100, regions=None,
                      source='r7'):
    """Plot cumulative trends per capita (pull data from global DFS dict).

    - lastday: up to this day.
    - source: 'r7' (7-day rolling average), 'raw' (no smoothing), 'sg'
      (Savitsky-Golay smoothed).
    - region_list: list of municipalities (including e.g. 'HR:Zuid',
      'POP:100-200').
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    # fig.subplots_adjust(top=0.945, bottom=0.085, left=0.09, right=0.83)


    for region in regions:
        df, npop = nl_regions.select_cases_region(DFS['cases'], region)
        df = df.iloc[-ndays:]
        ax.semilogy(df['Total_reported'] * (1e5/npop), label=region)

    ax.set_ylabel('Cumulatieve Covid-19 gevallen per 100k')
    tools.set_xaxis_dateformat(ax)
    ax.legend()
    fig.show()


def plot_anomalies_deltas(ndays=120):
    """Show effect of anomaly correction."""
    df, _npop = get_region_data('Nederland', correct_anomalies=True)
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 5))

    col_labs = [('Delta_orig', 'Raw'), ('Delta', 'Anomalies corrected')]
    for col, lab in col_labs:
        ax.semilogy(df.iloc[-ndays:][col], label=lab)
    ax.legend()
    tools.set_xaxis_dateformat(ax, maxticks=7)
    title = 'Anomaly correction'
    ax.set_title(title)
    fig.canvas.manager.set_window_title(title)
    fig.show()


def _add_mobility_data_to_R_plot(ax):

    try:
        df = get_g_mobility_data()
    except Exception as e:
        print(f'No Google Mobility data: {e.__class__.__name__}: {e}')
        return

    ymin, ymax = ax.get_ylim()
    y0 = ymin + (ymax - ymin)*0.85
    scale = (ymax - ymin)*0.1

    cols = ['retail_recr', 'transit', 'work', 'resid']
    # from rcParams['axes.prop_cycle']Ik werk met he
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 5
    scale /= df[cols].values.max()

    ts = df.index
    ax.axhline(y0, linestyle='--', color='gray')
    for c, clr in zip(cols, colors):
        hs = df[c].values
        # ax.fill_between(ts, y0-scale*hs, y0+scale*hs, color=clr, alpha=0.3, label=c)
        ax.plot(ts, y0+scale*hs, color=clr, label=c)



def _coord_format_Rplot(axR, axD, Tgen):
    """Setup cursor coordinate formatting for R graph, from ax and twinx ax.

    axR: R/date axis; axD: doubling time/date axis
    Tgen: generation time"""
    def format_coord(x, y):
        #display_coord = axR.transData.transform((x,y))

        #inv =
        # convert back to data coords with respect to ax
        # ax_coord = inv.transform(display_coord)  # x2, y2

        # In this case, (x1, x2) == (x2, y2), but the y2 labels
        # are custom made. Otherwise:
        # x2, y2 = axD.transData.inverted().transform(axR.transData.transform((x,y)))

        t, R = x, y
        from matplotlib.dates import num2date
        tm_str = num2date(t).strftime('%Y-%m-%d %H:%M')
        T2 = np.log(2)/np.log(R) * Tgen
        return f'{tm_str}: R={R:.3f}, T2={T2:.3g} d'
    axD.format_coord = format_coord



def plot_Rt(ndays=100, lastday=-1, delay='DEFAULT', regions='Nederland', source='r7',
            Tc=4.0, correct_anomalies=True, g_mobility=False, mode='show',
            ylim=None, only_trendlines=False):
    """Plot R number based on growth/shrink in daily cases.

    - lastday: use case data up to this day.
    - delay: assume delay days from infection to positive report.
      alternatively: list of (timestamp, delay) tuples if the delay varies over time.
      The timestamps refer to the date of report. See doc of estimeate_Rt_series.
      alternatively: 'DEFAULT' for default list.
    - source: 'r7' or 'sg' for rolling 7-day average or Savitsky-Golay-
      filtered data.
    - Tc: generation interval timepd.to_datetime(matplotlib.dates.num2date(ax.get_xlim()))
    - regions: comma-separated string (or list of str);
      'Nederland', 'V:xx' (holiday region), 'P:xx' (province), 'M:xx'
      (municipality).
      set to 'DUMMY' or '' to plot only RIVM curve.
    - correct_anomalies: whether to correct for known reporting anomalies.
    - g_mobility: include Google mobility data (experimental, not very usable yet).
    - mode: 'show' or 'return_fig'
    - ylim: optional y axis range (ymin, ymax)
    - only_trendlines: no scatter points, only trend lines.
    """

    Rt_rivm = DFS['Rt_rivm']

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.09, right=0.92)
    plt.xticks(rotation=-20)

    if ylim:
        ax.set_ylim(*ylim)

    # dict: municitpality -> population

    # from rcParams['axes.prop_cycle']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 5

    markers = 'o^v<>s+x123' * 5

    labels = [] # tuples (y, txt)
    if isinstance(regions, str):
        regions = regions.split(',')
    if len(regions) == 0:
        regions = ['DUMMY']


    for i_region, (region, color, marker) in enumerate(zip(regions, colors, markers)):

        df1, _npop = get_region_data(
            'Nederland' if region=='DUMMY' else region,
            lastday=lastday, correct_anomalies=correct_anomalies
            )
        source_col = dict(r7='Delta7r', sg='DeltaSG')[source]

        # skip the first 10 days because of zeros
        Rdf = estimate_Rt_df(df1[source_col].iloc[10:], delay=delay, Tc=Tc)
        Rt = Rdf['Rt'].iloc[-ndays:]
        delays = Rdf['delay'].iloc[-ndays:]
        delay_min, delay_max = delays.min(), delays.max()
        if delay_min == delay_max:
            delay_str = f'{delay_min:.2g}'
        else:
            delay_str = f'{delay_min:.2g}-{delay_max:.2g}'

        fmt = marker
        psize = 5 if ndays < 30 else 3

        if region.startswith('POP:'):
            label = region[4:] + ' k inw.'
        elif region == 'Nederland':
            label = 'R schatting Nederland'
        else:
            label = re.sub('^[A-Z]+:', '', region)

        if not only_trendlines and region != 'DUMMY':
            ax.plot(Rt[:-3], fmt, label=label, markersize=psize, color=color)
            ax.plot(Rt[-3:], fmt, markersize=psize, color=color, alpha=0.35)

        # add confidence range (ballpark estimate)
        print(region)

        # Last 3 days are extrapolation, but peek at one extra day for the
        # smooth curve generation.
        # SG filter (13, 2): n=13 (2 weeks) will iron out all weekday effects
        # remaining despite starting from a 7-day average.
        Rt_smooth = scipy.signal.savgol_filter(Rt.iloc[:-2], 13, 2)[:-1]
        Rt_smooth = pd.Series(Rt_smooth, index=Rt.index[:-3])
        print(f'Smooth R: {Rt_smooth.iloc[-1]:.3g} @ {Rt_smooth.index[-1]}')

        if region == 'Nederland':
            # Error: hardcoded estimate 0.05. Because of SG filter, last 6 days
            # are increasingly less accurate.
            Rt_err = np.full(len(Rt_smooth), 0.05)
            Rt_err[-6:] *= np.linspace(1, 1.4, 6)
            ax.fill_between(Rt_smooth.index,
                            Rt_smooth.values-Rt_err, Rt_smooth.values+Rt_err,
                            color=color, alpha=0.15, zorder=-10
                            )

            # This is for posting on Twitter
            Rt_smooth_latest = Rt_smooth.iloc[-1]
            Rt_point_latest = Rt.iloc[-4]
            date_latest = Rt.index[-4].strftime('%d %b')
            slope = (Rt_smooth.iloc[-1] - Rt_smooth.iloc[-4])/3
            if abs(Rt_smooth_latest - Rt_point_latest) < 0.015:
                txt = f'R={(Rt_smooth_latest+Rt_point_latest)/2:.2f}'
            else:
                txt = (f'R={Rt_point_latest:.2f} (datapunt), '
                       f'R={Rt_smooth_latest:.2f} (voorlopige trendlijn)')
            print(f'Update reproductiegetal Nederland t/m {date_latest}: {txt}.'
                  ' #COVID19NL\n'
                  f'Trend: {"+" if slope>=0 else "−"}{abs(slope):.3f} per dag.')


        label = None

        if region == 'Nederland':
            label = 'R trend Nederland'
        elif only_trendlines:
            label = re.sub('^.*:', '', region)

        if region != 'DUMMY':
            smooth_line = ax.plot(Rt_smooth[:-5], color=color, alpha=1, zorder=0,
                                  linestyle=('-' if i_region < 10 else '-.'),
                                  label=label
                                  )
            ax.plot(Rt_smooth[-6:], color=color, alpha=1, zorder=0,
                    linestyle='--', dashes=(2,2))

            labels.append((Rt[-1], f' {label}'))

    if len(labels) == 0:
        print('Note: no regions to plot.')

    if Rt_rivm is not None:
        tm_lo, tm_hi = Rt.index[[0, -1]] # lowest timestamp
        tm_rivm_est = Rt_rivm[Rt_rivm['R'].isna()].index[0] # 1st index with NaN
        # final values
        df_Rt_rivm_final = Rt_rivm.loc[tm_lo:tm_rivm_est, ['R', 'Rt_update']]
        ax.plot(df_Rt_rivm_final.iloc[:-1]['R'], 'k-', label='RIVM')
        ax.plot(df_Rt_rivm_final.iloc[:-1]['Rt_update'], 'k^', markersize=4,
                label='RIVM updates', zorder=10)
        # estimates
        Rt_rivm_est = Rt_rivm.loc[tm_rivm_est-pd.Timedelta(1, 'd'):Rt.index[-1]]
        # print(Rt_rivm_est)
        ax.fill_between(Rt_rivm_est.index, Rt_rivm_est['Rmin'], Rt_rivm_est['Rmax'],
                        color='k', alpha=0.15, label='RIVM prognose')

    iex = dict(r7=3, sg=7)[source] # days of extrapolation

    # add_labels(ax, labels, lab_x)
    # marker at 12:00 on final day (index may be a few hours off)
    t_mark = Rt.index[-iex-1]
    t_mark += pd.Timedelta(12-t_mark.hour, 'h')
    ax.axvline(t_mark, color='gray')
    ax.axhline(1, color='k', linestyle='--')
    ax.text(t_mark, ax.get_ylim()[1], Rt.index[-4].strftime("%d %b "),
            rotation=90, horizontalalignment='right', verticalalignment='top')

    xnotes = []
    if source != 'r7':
        xnotes.append(source)
    if correct_anomalies:
        dfa = DFS['anomalies'].copy()
        dfa = dfa.loc[dfa['fraction'] > 0]
        anom_date = dfa.index[-1].strftime('%d %b') # most recent anomaly date
        xnotes.append(f'correctie pos. tests o.a. {anom_date}')
    if xnotes:
        xnotes = ", ".join([""]+xnotes)
    else:
        xnotes = ''

    ax.set_title(f'Reproductiegetal o.b.v. positieve tests; laatste {iex} dagen zijn een extrapolatie\n'
                 f'(Generatie-interval: {Tc:.3g} dg, rapportagevertraging {delay_str} dg{xnotes})'
                 )
    ax.set_ylabel('Reproductiegetal $R_t$')

    # setup the x axis before adding y2 axis.
    tools.set_xaxis_dateformat(ax, maxticks=10)

    # get second y axis with doubling times.
    # Adjust values shown depending on scale range.
    ax2 = ax.twinx()
    _coord_format_Rplot(ax, ax2, Tgen=Tc)
    T2s = np.array([
        -60, -21, -14, -10, -7, -4, -3,
        9999,
        60, 21, 14, 10, 7, 4, 3, 2.5, 2.2, 2, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3
        ])
    Rmax = ax.get_ylim()[1] # highest R value
    if Rmax > 2:
        T2s = T2s[(np.abs(T2s) < 10) | (T2s==9999)]
    T2s = np.concatenate(([], T2s))

    y2ticks = 2**(Tc/T2s)
    y2labels = [f'{t2 if t2 != 9999 else "∞"}' for t2 in T2s]
    ax2.set_yticks(y2ticks)
    ax2.set_yticklabels(y2labels)
    ax2.set_ylim(*ax.get_ylim())
    ax2.set_ylabel('Halverings-/verdubbelingstijd (dagen)')

    xlim = (Rt.index[0] - pd.Timedelta('12 h'), Rt.index[-1] + pd.Timedelta('3 d'))
    ax.set_xlim(*xlim)
    _add_event_labels(ax, Rt.index[0], Rt.index[-1], flagmatch='RGraph')
    if g_mobility:
        _add_mobility_data_to_R_plot(ax)

    ax.text(0.99, 0.98, '@hk_nien', transform=ax.transAxes,
            va='top', ha='right', rotation=90)

    ax.legend(loc='upper left')

    if mode == 'show':
        fig.canvas.manager.set_window_title(f'Rt ({", ".join(regions)[:30]}, ndays={ndays})')
        fig.show()



    elif mode == 'return_fig':
        return fig

    else:
        raise ValueError(f'mode={mode!r}')


def plot_Rt_oscillation():
    """Uses global DFS['Rt_rivm'] variable."""


    fig, axs = plt.subplots(2, 1, tight_layout=True)

    df_Rt_rivm = DFS['Rt_rivm']

    series_Rr = df_Rt_rivm['R'][~df_Rt_rivm['R'].isna()].iloc[-120:]
    Rr = series_Rr.to_numpy()
    Rr_smooth = scipy.signal.savgol_filter(Rr, 15, 2)

    dates = series_Rr.index
    ax = axs[0]
    n = len(Rr)
    ax.plot(dates, Rr, label='Rt (RIVM)')
    ax.plot(dates, Rr_smooth, label='Rt smooth', zorder=-1)
    ax.plot(dates, Rr - Rr_smooth, label='Difference')
    ax.set_ylabel('R')
    ax.set_xlim(dates[0], dates[-1])
    tools.set_xaxis_dateformat(ax)
    plt.xticks(rotation=0) # undo; doesn't work with subplots
    ax.legend()
    ax = axs[1]
    # window = 1 - np.linspace(-1, 1, len(Rr))**2


    window = scipy.signal.windows.tukey(n, alpha=(n-14)/n)
    n_padded = n*3//7*7 # make sure it's a multiple of 1 week
    spectrum = np.fft.rfft((Rr-Rr_smooth) * window, n=n_padded)
    freqs = 7/n_padded * np.arange(len(spectrum))
    mask = (freqs < 2.4)
    ax.plot(freqs[mask], np.abs(spectrum[mask])**2)
    ax.set_xlabel('Frequency (1/wk)')
    ax.set_ylabel('Power')
    ax.grid()

    fig.canvas.manager.set_window_title('Rt oscillation')

    fig.show()


def reset_plots():
    """Close plots and adjust default matplotlib settings."""

    # Note: need to run this twice before NL locale takes effect.
    try:
        locale.setlocale(locale.LC_ALL, 'nl_NL.UTF-8')
    except locale.Error as e:
        print(f'Warning: cannot set language: {e.args[0]}')
    plt.rcParams["date.autoformatter.day"] = "%d %b"
    plt.close('all')


def plot_barchart_daily_counts(istart=-70, istop=None, region='Nederland', figsize=(10, 4)):
    """Plot daily case counts and corrections (from anomalies).

    Parameters: iloc range; default .iloc[-70:].
    """
    df_full, population = get_region_data(region)

    # set index time values to noon, not 10:00
    df_full.index = df_full.index + (12 - df_full.index.hour) * pd.Timedelta('1 h')
    df = df_full.iloc[istart:istop]
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    width = pd.Timedelta('24 h')
    mask_weekend = df.index.dayofweek.isin((5, 6))

    plot_labels = []
    def addplot(func, *args, **kwargs):
        """Add plot, store label in plot_labels."""
        func(*args, **kwargs)
        if 'label' in kwargs:
            plot_labels.append(kwargs['label'])

    edgecolor = '#888888'
    edge_width = min(100 / len(df), 1.5)
    addplot(
        ax.bar, df.index, df['Delta_orig']*population,
        width=width, label='Positief per dag', color='#77aaff',
        edgecolor=edgecolor, lw=edge_width
        )
    ax.bar(df.index[mask_weekend], df.loc[mask_weekend, 'Delta_orig']*population,
           width=width, color='black', alpha=0.25)
    mask = (df['Delta'] != df['Delta_orig'])
    if mask.sum() > 0:
        addplot(
            ax.bar, df.index[mask], df.loc[mask, 'Delta']*population,
            width=width, alpha=0.5, color='#ff7f0e',
            label='Schatting i.v.m. datastoring'
            )
    addplot(
        ax.plot, df.index, df['Delta_dowc']*population, 'x',
        color='#aa0000',
        markersize=4*min(5, max(1, 100/len(df))),
        zorder=10,
        label='Weekdageffect gecorrigeerd'
        )

    idx_3d = df_full.index[-4]
    addplot(
        ax.plot, df.loc[df.index <= idx_3d, 'Delta7r']*population,
        label='7d-gemiddelde', color='red', lw=2
        )
    # generate white/black dashed line for better contrast
    ax.plot(
        df.loc[df.index >= idx_3d, 'Delta7r']*population,
        color='white', lw=2
        )
    addplot(
        ax.plot, df.loc[df.index >= idx_3d, 'Delta7r']*population,
        label='7d schatting',
        linestyle=(2, (2, 1)), color='#000000', lw=2
        )

    #mask = df.index.dayofweek == 3
    #ax.plot(df.index[mask], df.loc[mask, 'Delta_orig']*population,
    #        'g^', markersize=8, label='Donderdagen')

    # ensure legend in same order as plots.
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        [handles[labels.index(lab)] for lab in plot_labels], plot_labels,
        loc='best', framealpha=1
        )
    leg.set_zorder(20)
    ax.set_yscale('log')
    ax.set_ylabel('Positieve gevallen per dag')
    tools.set_xaxis_dateformat(ax)
    ax.grid(which='minor', axis='y')
    tools.set_yaxis_log_minor_labels(ax)

    title = f'Positieve tests per dag ({region}) - log schaal, rechte lijn = exponentiële groei'
    ax.set_title(title)
    fig.canvas.manager.set_window_title(title)
    fig.show()


if __name__ == '__main__':

    print('Please run nlcovidstats_show.py or plot_R_from_daily_cases.py')
