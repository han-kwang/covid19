#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Country data of B.1.1.7 occurrence.

Function: get_country_data().

@author: @hk_nien
"""
import re
from pathlib import Path
import pandas as pd
import datetime
import numpy as np

def _ywd2date(ywd):
    """Convert 'yyyy-Www-d' string to date (12:00 on that day)."""

    twelvehours = pd.Timedelta('12 h')

    dt = datetime.datetime.strptime(ywd, "%G-W%V-%w") + twelvehours
    return dt


def _add_odds_column(df):

    df['or_b117'] = df['f_b117'] / (1 + df['f_b117'])

def _convert_ywd_records(records, colnames=('f_b117',)):
    """From records to DataFrame with columns.

    Records are tuples with ('yyyy-Www-d', value, ...).
    """

    df = pd.DataFrame.from_records(records, columns=('Date',) + tuple(colnames))
    df['Date'] = [_ywd2date(r) for r in df['Date']]
    df = df.set_index('Date')
    if 'f_b117' in colnames  and 'or_b117' not in colnames:
        _add_odds_column(df)

    return df


def _get_data_uk_genomic():

    #https://twitter.com/hk_nien/status/1344937884898488322
    # data points read from plot (as ln prevalence)
    seedata = {
        '2020-12-21': [
            ['2020-09-25', -4.2*1.25],
            ['2020-10-02', -3.5*1.25],
            ['2020-10-15', -3.2*1.25],
            ['2020-10-20', -2.3*1.25],
            ['2020-10-29', -2.3*1.25],
            ['2020-11-05', -1.5*1.25],
            ['2020-11-12', -0.9*1.25],
            ['2020-11-19', -0.15*1.25],
            ['2020-11-27', 0.8*1.25]
            ],
        '2020-12-31': [
            ['2020-10-31', -2.1],
            ['2020-11-08', -1.35],
            ['2020-11-15', -0.75],
            ['2020-11-22', -0.05],
            ['2020-11-29', 0.05],
            ]
        }

    cdict = {}
    for report_date, records in seedata.items():
        df = pd.DataFrame.from_records(records, columns=['Date', 'ln_odds'])
        df['Date'] = pd.to_datetime(df['Date'])
        odds = np.exp(df['ln_odds'])
        df['f_b117'] = odds / (1 + odds)
        df = df[['Date', 'f_b117']].set_index('Date')

        cdict[f'UK (South East, sequenced, {report_date})'] = df

    return cdict



def _get_data_countries_weeknos():
    """Countries with f_117 data by week number."""

    country_records = {
        # https://covid19.ssi.dk/-/media/cdn/files/opdaterede-data-paa-ny-engelsk-virusvariant-sarscov2-cluster-b117--01012021.pdf?la=da
        'DK (1 jan)': [
            # Year-week-da, n_pos, f_b117
            ('2020-W49-4', 0.002),
            ('2020-W50-4', 0.005),
            ('2020-W51-4', 0.009),
            ('2020-W52-4', 0.023)
            ],
        # https://www.covid19genomics.dk/statistics
        'DK (19 jan)': [
            # Year-week-da, n_pos, f_b117
            ('2020-W48-4', 0.002),
            ('2020-W49-4', 0.002),
            ('2020-W50-4', 0.004),
            ('2020-W51-4', 0.008),
            ('2020-W52-4', 0.020),
            ('2020-W53-4', 0.024),
            ('2021-W01-4', 0.039),
            ('2021-W02-4', 0.070), # preliminary# preliminary
            ],

        # OMT advies #96
        # https://www.tweedekamer.nl/kamerstukken/brieven_regering/detail?id=2021Z00794&did=2021D02016
        # https://www.rivm.nl/coronavirus-covid-19/omt (?)

        'NL OMT-advies #96': [
            # Year-week-da, n_pos, f_b117
            ('2020-W49-4', 0.011),
            ('2020-W50-4', 0.007),
            ('2020-W51-4', 0.011),
            ('2020-W52-4', 0.014),
            ('2020-W53-4', 0.052),
            ('2021-W01-4', 0.119), # preliminary
            ],

        # https://www.tweedekamer.nl/sites/default/files/atoms/files/20210120_technische_briefing_commissie_vws_presentati_jaapvandissel_rivm_0.pdf
        'NL RIVM 2021-01-20': [
            # Year-week-da, n_pos, f_b117
            ('2020-W49-5', 0.015),
            ('2020-W50-5', 0.010),
            ('2020-W51-5', 0.015),
            ('2020-W52-5', 0.020),
            ('2020-W53-5', 0.050),
            ('2021-W01-5', 0.090), # preliminary
            ],
        # https://virological.org/t/tracking-sars-cov-2-voc-202012-01-lineage-b-1-1-7-dissemination-in-portugal-insights-from-nationwide-rt-pcr-spike-gene-drop-out-data/600
        'PT 19 jan': [
            ('2020-W49-4', 0.019),
            ('2020-W50-4', 0.009),
            ('2020-W51-4', 0.013),
            ('2020-W52-4', 0.019),
            ('2020-W53-4', 0.032),
            ('2021-W01-4', 0.074),
            ('2021-W02-4', 0.133),
            ]
        }

    cdict = {}
    for desc, records in country_records.items():
        cdict[desc] = _convert_ywd_records(records, ['f_b117'])

    return cdict

regions_pop = {
    'South East England': 9180135,
    'London': 8961989,
    'North West England': 7341196,
    'East England': 6236072,
    'West Midlands': 5934037,
    'South West England': 5624696,
    'Yorkshire': 5502967,
    'East Midlands': 4835928,
    'North East England': 2669941,
    }

regions_pop['multiple regions'] = sum(regions_pop.values())

def _get_data_uk_regions(subtract_bg=True):
    """Get datasets for UK regions. Original data represents 'positive population'.
    Dividing by 28 days and time-shifting 14 days to get estimated daily increments.

    With subtract_bg: Subtracting lowest UK-region value - assuming background
    false-positive for S-gene target failure.

    Data source: Walker et al., https://doi.org/10.1101/2021.01.13.21249721
    """

    index_combi = pd.date_range('2020-09-28', '2020-12-14', freq='7 d')
    df_combi = pd.DataFrame(index=index_combi)
    ncolumns = ['pct_sgtf', 'pct_othr']
    for col in ncolumns:
        df_combi[col] = 0

    cdict = {'UK (multiple regions)': df_combi}
    for fpath in sorted(Path('data').glob('uk_*_b117_pop.csv')):
        ma = re.search('uk_(.*)_b117', str(fpath))
        region = ma.group(1).replace('_', ' ')
        df = pd.read_csv(fpath, comment='#').rename(columns={'Unnamed: 0': 'Date'})

        df['Date'] = pd.to_datetime(df['Date']) - pd.Timedelta(14, 'd')
        df = df.set_index('Date')

        # interpolate and add to the combined dataframe.
        df2 = pd.DataFrame(index=index_combi) # resampling data here
        df2 = df2.merge(df[ncolumns], how='outer', left_index=True, right_index=True)
        df2 = df2.interpolate(method='quadratic').loc[index_combi]

        for col in ncolumns:
            df_combi[col] += df2[col]

        cdict[f'UK ({region})'] = df

    # convert to estimated new cases per day.
    for key, df in cdict.items():
        region = re.search(r' \((.*)\)', key).group(1)


        # estimate false-positive for SGTF as representing B.1.1.7
        if subtract_bg:
            pct_bg = df['pct_sgtf'].min()
        else:
            pct_bg = 0.0

        df['n_b117'] = ((df['pct_sgtf'] - pct_bg)*(0.01/28 * regions_pop[region])).astype(int)
        df['n_oth'] = ((df['pct_othr'] + pct_bg)*(0.01/28 * regions_pop[region])).astype(int)


        # this doesn't work
        # if subtract_bg:
        #     pct_tot = df['pct_sgtf'] + df['pct_othr']
        #     # f: fraction of positive test. Correct for background.
        #     f_sgtf = df['pct_sgtf']/pct_tot
        #     f_sgtf_min = f_sgtf.min()
        #     f_sgtf -= f_sgtf_min

        #     # convert back to pct values
        #     df['pct_sgtf'] = pct_tot * f_sgtf
        #     df['pct_othr'] = pct_tot * (1-f_sgtf)
        # df['n_b117'] = (df['pct_sgtf'] * (0.01/28 * regions_pop[region])).astype(int)
        # df['n_oth'] = (df['pct_othr'] * (0.01/28 * regions_pop[region])).astype(int)

        df.drop(index=df.index[df['n_b117'] <= 0], inplace=True)

        df['n_pos'] = df['n_b117'] + df['n_oth']
        df['or_b117'] = df['n_b117'] / df['n_oth']
        df['f_b117'] = df['or_b117']/(1 + df['or_b117'])

    for col in ncolumns + ['n_pos']:
        df_combi[col] = np.around(df_combi[col], 0).astype(int)

    return cdict




def get_data_countries(recent_only=True, uk_regions=True, subtract_uk_bg=True):
    """Return dict, key=description, value=dataframe with Date, n_pos, f_b117, or_b117."""


    cdict_uk = _get_data_uk_regions(subtract_bg=subtract_uk_bg)
    key = 'UK (multiple regions)'
    cdict_uk_2 = {key: cdict_uk[key]} # selection

    if uk_regions:
        for k, v in cdict_uk.items():
            if k in cdict_uk_2:
                continue
            # rename UK (x) -> x
            m = re.search(r'UK \((.*)\)', k)
            cdict_uk_2[f'{m.group(1)} (UK)'] = v

    cdict = {
        **cdict_uk_2,
        **_get_data_uk_genomic(), # this is now covered by cdict_uk
        **_get_data_countries_weeknos(),
        }

    if recent_only:
        # only the last one of each country code
        seen = set()
        for key in list(cdict.keys())[::-1]:
            cc = key[:2]
            if re.match('[A-Z][A-Z]', cc) and cc in seen:
                del cdict[key]
            else:
                seen.add(cc)

    return cdict
