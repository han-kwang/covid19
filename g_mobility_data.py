#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Get/preprocess Google mobility data for the Netherlands.

Created on Sat Feb  6 21:31:20 2021

@author: @hk_nien
"""

import zipfile
import io
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import tools


def download_g_mobility_data():
    """Download Google Mobility data, write to data/2020_NL_⋯.csv."""

    url = 'https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip'
    print(f'Getting Google Mobility data ...')
    with urllib.request.urlopen(url) as response:
        data_bytes = response.read()
    zipf = zipfile.ZipFile(io.BytesIO(data_bytes))
    fname = zipf.extract('2020_NL_Region_Mobility_Report.csv', path='data/')
    print(f'Wrote {fname} .')


def get_g_mobility_data():
    """Return dataframe with simplified mobility data for the Netherlands.

    Index is timestamp.

    Values are 1.0 for baseline = (1 + deviation/100)
    from the original deviations. Smoothed to remove weekday effects.

    Column names abbreviated: retail_recr, groc_phar, parks, transit, work, resid.
    """

    df = pd.read_csv('data/2020_NL_Region_Mobility_Report.csv')
    df = df.loc[df['sub_region_1'].isna()]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    df.rename(columns={
        'retail_and_recreation_percent_change_from_baseline': 'retail_recr',
        'grocery_and_pharmacy_percent_change_from_baseline': 'groc_phar',
        'parks_percent_change_from_baseline': 'parks',
        'transit_stations_percent_change_from_baseline': 'transit',
        'workplaces_percent_change_from_baseline': 'work',
        'residential_percent_change_from_baseline': 'resid'
        }, inplace=True)

    df.drop(columns=['country_region_code', 'country_region', 'sub_region_1', 'sub_region_2',
           'metro_area', 'iso_3166_2_code', 'census_fips_code'], inplace=True)

    for c in df.columns:
        smooth_data = scipy.signal.savgol_filter(df[c].values, 13, 2, mode='interp')
        df[c] = 1 + 0.01 * smooth_data

    # check whether it's up to date.
    # Mobility data released on 2nd or 3rd of the month?
    today = pd.to_datetime('now')
    if today.month != (df.index[-1].month % 12 + 1) and today.day >= 3:
        print('Google mobility report may be outdated. Call download_g_mobility_data().')

    return df


if __name__ == '__main__':

    df = get_g_mobility_data()
    plt.close('all')
    fig, ax = plt.subplots(tight_layout=True, figsize=(14, 6))
    for c in df.columns:
        ax.plot(df.index, df[c], label=c)
    ax.legend()
    ax.grid()
    tools.set_xaxis_dateformat(ax, maxticks=25)
    fig.show()

