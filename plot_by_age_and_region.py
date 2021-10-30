#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 13:43:12 2021

@author: @hk_nien
"""
from multiprocessing import Pool, cpu_count
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from nl_regions import get_holiday_regions_by_ggd
import nlcovidstats as nlcs
import casus_analysis as ca
from tools import set_xaxis_dateformat

def invert_mapping(m):
    """Invert mapping k->[v0, ...] to Series with v->k."""
    mi = {}
    for k, vs in m.items():
        for v in vs:
            mi[v] = k
    return pd.Series(mi)

def _get_summary_1date(date):
    """Return summary dict for casus data for specified date (yyyy-mm-dd)."""
    print('.', end='', flush=True)
    df = ca.load_casus_data(date)

    # invert mapping
    regions_ggd2hol = invert_mapping(get_holiday_regions_by_ggd())
    age_group_agg = invert_mapping({
         '0-19': ['0-9', '10-19'],
         '20+': ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+'],
         'Other': ['<50', 'Unknown']
        })
    df['Holiday_region'] = regions_ggd2hol[df['Municipal_health_service']].values
    missing = df['Holiday_region'].isna()
    if np.any(missing):
        missing = df.loc[missing, 'Municipal_health_service'].unique()
        msg = f'Unrecognized municipal health service {missing}'
        raise KeyError(msg)

    df['Agegroup_agg'] = age_group_agg[df['Agegroup']].values
    summary = {'Date_file': df.iloc[-1]['Date_file']}
    for (hr, aga), df1 in df.groupby(['Holiday_region', 'Agegroup_agg']):
        summary[f'{hr}:{aga}'] = len(df1)
    summary = {k: summary[k] for k in sorted(summary)}
    return summary


def get_summary_hr_age_by_date_range(
        date_a='2020-07-01', date_b='2099-01-01',
        csv_fn='data/cum_cases_by_holiday_region_and_age.csv'
        ):
    """Return summary DataFrame for date range. Parallel processing.

    Parameters:

    - date_a: lowest date ('yyyy-mm-dd')
    - date_b: highest date ('yyyy-mm-dd')
        (or far in the future to use whatever available).
    - csv_fn: optional csv filename to load data from. Data not in CSV
      will be regenerated (slow). File will be updated with any new data.
      Set to None to skip.
      If set but file is nonexistent, a new file will be created.

    Return DataFrame:

    - index: Date_file as Timestamp (0:00:00 time of day), for file Date.
    - columns: (Midden|Noord|Zuid):(0-19|20+|Other)'.
      Values: total number of cases up to that date.
    """

    date_a, date_b = [pd.to_datetime(d) for d in [date_a, date_b]]
    if date_a < pd.to_datetime('2020-07-01'):
        raise ValueError(f'date_a {date_a}: no data available.')

    if csv_fn:
        try:
            df_cached = pd.read_csv(csv_fn)
            df_cached['Date_file'] = pd.to_datetime(df_cached['Date_file'])
            df_cached.set_index('Date_file', inplace=True)
        except FileNotFoundError:
            print(f'Warning: no csv file {csv_fn!r}; will create new.')
            df_cached = pd.DataFrame()

    dates = []
    date = date_a
    while date <= date_b:
        if date not in df_cached.index:
            try:
                ca._find_casus_fpath(date.strftime('%Y-%m-%d'))
            except FileNotFoundError:
                break
            dates.append(date)
        date += pd.Timedelta(1, 'd')

    if len(dates) > 0:
        print(f'Processing casus data for {len(dates)} dates. This may take a while.')
        # shuffle order so that the progress indicator doesn't slow down
        # towards the end when the data is large.
        random.seed(1)
        random.shuffle(dates)
        ncpus = cpu_count()
        print(f'({ncpus} workers)', end='', flush=True)
        with Pool(ncpus) as p:
            summaries = p.map(_get_summary_1date, dates)
        print('done.')
        df_new = pd.DataFrame.from_records(summaries).set_index('Date_file')
    else:
        df_new = pd.DataFrame()

    if csv_fn:
        if len(df_new) > 0:
            df_cached = df_cached.append(df_new).sort_index()
            df_cached.to_csv(csv_fn)
            print(f'Updated {csv_fn} .')
        row_select = (df_cached.index >= date_a) & (df_cached.index <= date_b)
        df_new = df_cached.loc[row_select]

    return df_new


def plot_growth_factors_by_age_and_holiday_region(
        date_a, date_b='2099-01-01', time_shift=-7.5, gf_range=(0, 2.0),
        estimate_last3=False
        ):
    """Plot 7d growth factors by age and holiday region, and the ratio children/adults.

    The purpose is to get an idea of the impact of holidays.
    This uses casus datasets, which must be available. Preprocessing the casus
    datasets is very slow; preprocessed output will be saved as
    data/cum_cases_by_holiday_region_and_age.csv , which will be reused
    on future calls.

    Parameters:

    - date_a: lowest date ('yyyy-mm-dd')
    - date_b: highest date ('yyyy-mm-dd')
        (or far in the future to use whatever available).
    - time_shift: negative time interval (in days) to convert growth-factor
      timestamps to approximate date of infection.
      Default -7.5 = -(3.5 + 4), where 3.5 handles the growth-factor interval
      (7 days) and 4 an additional delay to account for the date of infection.
    - gf_range: y range on growth-factor plot.
    - estimate_last3: True to get 3 extra estimated days of 7d-average.
      This tends to be noisy.
    """

    df = get_summary_hr_age_by_date_range(date_a, date_b)
    df_delta = df.diff().iloc[1:]
    df_delta = df_delta[[c for c in df.columns if not 'Other' in c]]

    df_delta7 = df_delta.rolling(7, center=True).mean()

    # estimate weekday correction and apply to last 3 missing values.
    if estimate_last3:
        n_weeks = 1
        for i_dow in range(3):
            row_select = slice(-1-i_dow-n_weeks*7, -1-i_dow, 7)
            f = (df_delta7.iloc[row_select] / df_delta.iloc[row_select]).mean(axis=0)
            df_delta7.iloc[-1-i_dow] = df_delta.iloc[-1-i_dow] * f

    # df_delta7 = np.around(df_delta7, 1)

    df_gf = df_delta7.iloc[7:] / df_delta7.iloc[:-7].values
    if estimate_last3:
        df_gf = df_gf.iloc[3:]
    else:
        df_gf = df_gf.iloc[3:-4]

    df_gf.index = df_gf.index + pd.Timedelta(time_shift, 'd')

    fig, (ax_gf, ax_gfr) = plt.subplots(
        2, 1, figsize=(9, 8), tight_layout=True,
        sharex=True
        )
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = '^ov'
    for hr, color, marker in zip(['Noord', 'Midden', 'Zuid'], colors, markers):
        for aga, lsty in zip(['0-19', '20+'], ['-', '--']):
            ax_gf.plot(df_gf[f'{hr}:{aga}'], color=color, linestyle=lsty,
                       label=f'{hr}: {aga}')
        ratios = df_gf[f'{hr}:0-19'] / df_gf[f'{hr}:20+']
        ratios_s = savgol_filter(ratios.values, 13, 2, mode='interp')
        ax_gfr.plot(ratios.index, ratios, marker, markersize=3, color=color, label=hr)
        ax_gfr.plot(ratios.index, ratios_s, color=color)
    ax_gf.legend(loc='upper left')
    ax_gfr.legend(loc='upper left')
    ax_gf.set_title(
        "Groeifactor positieve tests per 7 dagen (7d-gemiddelde); "
        "vakantieregio's en leeftijdsgroepen\n"
        f'Verschoven met {-time_shift} dagen om terug te rekenen naar datum infectie.'
        )
    ax_gfr.set_title('Verhouding groeifactor leeftijd 0-19 t.o.v. 20+')
    ax_gf.set_ylim(*gf_range)
    ax_gf.grid(axis='y')
    set_xaxis_dateformat(ax_gf)
    set_xaxis_dateformat(ax_gfr, 'Datum infectie')

    # add event labels
    nlcs.init_data(autoupdate=False)
    nlcs._add_event_labels(ax_gf, df_gf.index[0], df_gf.index[-1],
                           with_ribbons=True, textbox=False, bottom=True,
                           flagmatch='RGraph')


    fig.show()


if __name__ == '__main__':
    plt.close('all')

    plot_growth_factors_by_age_and_holiday_region(
        '2021-06-01', '2099-01-01', time_shift=-7.5
        )
