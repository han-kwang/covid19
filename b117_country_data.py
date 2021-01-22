#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Country data of B.1.1.7 occurrence.

Function: get_country_data().

@author: @hk_nien
"""
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


def _get_data_uk():

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
        cdict[f'UK (SEE region, {report_date})'] = df

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


def get_data_countries():
    """Return dict, key=description, value=dataframe with Date, n_pos, f_b117, or_b117."""

    cdict = {
        **_get_data_uk(),
        **_get_data_countries_weeknos(),
        }

    return cdict
