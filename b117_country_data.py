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

    # https://twitter.com/hk_nien/status/1344937884898488322

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
            # https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-12-31-COVID19-Report-42-Preprint-VOC.pdf
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

        cdict[f'South East England (seq, {report_date})'] = df

    return cdict



def _get_data_countries_weeknos():
    """Countries with f_117 data by week number."""

    country_records = {
        # https://covid19.ssi.dk/-/media/cdn/files/opdaterede-data-paa-ny-engelsk-virusvariant-sarscov2-cluster-b117--01012021.pdf?la=da
        'DK (seq; 2021-01-01)': [
            # Year-week-da, n_pos, f_b117
            ('2020-W49-4', 0.002),
            ('2020-W50-4', 0.005),
            ('2020-W51-4', 0.009),
            ('2020-W52-4', 0.023)
            ],
        # https://www.covid19genomics.dk/statistics
        'DK (seq; 2021-01-26)': [
            # Year-week-da, n_pos, f_b117
            ('2020-W48-4', 0.002),
            ('2020-W49-4', 0.002),
            ('2020-W50-4', 0.004),
            ('2020-W51-4', 0.008),
            ('2020-W52-4', 0.020),
            ('2020-W53-4', 0.024),
            ('2021-W01-4', 0.040),
            ('2021-W02-4', 0.074),
            ('2021-W03-4', 0.121), # preliminary# preliminary
            ],

        # OMT advies #96
        # https://www.tweedekamer.nl/kamerstukken/brieven_regering/detail?id=2021Z00794&did=2021D02016
        # https://www.rivm.nl/coronavirus-covid-19/omt (?)

        'NL (seq; 2021-01-19; OMT)': [
            # Year-week-da, n_pos, f_b117
            ('2020-W49-4', 0.011),
            ('2020-W50-4', 0.007),
            ('2020-W51-4', 0.011),
            ('2020-W52-4', 0.014),
            ('2020-W53-4', 0.052),
            ('2021-W01-4', 0.119), # preliminary / calculation error (0.135???)
            ],

        # https://www.tweedekamer.nl/sites/default/files/atoms/files/20210120_technische_briefing_commissie_vws_presentati_jaapvandissel_rivm_0.pdf
        'NL (seq; 2021-01-20)': [
            # Year-week-da, n_pos, f_b117
            ('2020-W49-5', 0.015),
            ('2020-W50-5', 0.010),
            ('2020-W51-5', 0.015),
            ('2020-W52-5', 0.020),
            ('2020-W53-5', 0.050),
            ('2021-W01-5', 0.090), # preliminary
            ],
        # https://virological.org/t/tracking-sars-cov-2-voc-202012-01-lineage-b-1-1-7-dissemination-in-portugal-insights-from-nationwide-rt-pcr-spike-gene-drop-out-data/600
        'PT (seq; 2021-01-19)': [
            ('2020-W49-4', 0.019),
            ('2020-W50-4', 0.009),
            ('2020-W51-4', 0.013),
            ('2020-W52-4', 0.019),
            ('2020-W53-4', 0.032),
            ('2021-W01-4', 0.074),
            ('2021-W02-4', 0.133),
            ],
        # https://sciencetaskforce.ch/nextstrain-phylogentische-analysen/
        'CH (seq; 2021-01-26)': [
            ('2020-W51-4', 0.0004),
            ('2020-W52-4', 0.0043),
            ('2020-W53-4', 0.0074),
            ('2021-W01-4', 0.0153),
            ('2021-W02-4', 0.0329),
            ('2021-W03-4', 0.0989),
            ],
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
regions_pop['England (multiple regions; 2021-01-15)'] = sum(regions_pop.values())

uk_countries_pop = {
    'England': 56286961,
    'Scotland': 5463300,
    'Wales': 3152879,
    'Northern Ireland': 1893667,
    }


def _get_data_England_regions(subtract_bg=True):
    """Get datasets for England regions. Original data represents 'positive population'.
    Dividing by 28 days and time-shifting 14 days to get estimated daily increments.

    With subtract_bg: Subtracting lowest region value - assuming background
    false-positive for S-gene target failure.

    Data source: Walker et al., https://doi.org/10.1101/2021.01.13.21249721
    Published 2021-01-15.
    """

    index_combi = pd.date_range('2020-09-28', '2020-12-14', freq='7 d')
    df_combi = pd.DataFrame(index=index_combi)
    ncolumns = ['pct_sgtf', 'pct_othr']
    for col in ncolumns:
        df_combi[col] = 0

    cdict = {'England (SGTF; multiple regions; 2021-01-15)': df_combi}
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

        cdict[f'{region} (SGTF; 2021-01-15)'] = df

    # convert to estimated new cases per day.
    for key, df in cdict.items():

        region = re.match(r'(.*) \(.*', key).group(1)
        if region == 'England':
            region = 'England (multiple regions; 2021-01-15)'

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



def load_uk_ons_gov_country_by_var():
    """Return DataFrame based on data/ons_gov_uk_country_by_var.xlsx.

    Also return date string for publication date.

    index: Date.
    columns: {country_name}:{suffix}

    with suffix = 'pnew', 'pnew_lo', 'pnew_hi', 'poth', ..., 'pnid', ...

    for percentages new UK variant, CI low, CI high,
    other variant, not-identified.

    Data source:

    - https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/conditionsanddiseases/bulletins/coronaviruscovid19infectionsurveypilot/29january2021#positive-tests-that-are-compatible-with-the-new-uk-variant
    - https://www.ons.gov.uk/visualisations/dvc1163/countrybyvar/datadownload.xlsx
    """

    # Excel sheet: groups of 9 columns by country (England, Wales, NI, Scotland).
    xls_fname = 'data/ons_gov_uk_country_by_var.xlsx'

    # 1st round: sanity check and construct better column names.
    df = pd.read_excel(xls_fname, skiprows=3)
    assert np.all(df.columns[[1, 10]] == ['England', 'Wales'])
    assert df.iloc[0][0] == 'Date'
    assert df.iloc[0][1] == '% testing positive new variant compatible*'

    # percentages new variant, other, unidentified, with 95% CIs.
    col_suffixes = ['pnew', 'pnew_hi', 'pnew_lo', 'poth', 'poth_hi', 'poth_lo',
                    'pnid', 'pnid_hi', 'pnid_lo']

    colmap = {df.columns[0]: 'Date'}
    for i in range(1, 37, 9):
        country_name = df.columns[i]
        for j in range(9):
            colmap[df.columns[i+j]] = f'{country_name}:{col_suffixes[j]}'

    df.rename(columns=colmap, inplace=True)
    df.drop(columns=df.columns[37:], inplace=True)
    # find the end of the data
    i_stop = 2 + np.argmax(df['Date'].iloc[2:].isna())
    assert i_stop >= 44
    df = df.iloc[2:i_stop]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    if df.index[-1] == pd.to_datetime('2021-01-23'):
        date_pub = '2021-01-29'
    else:
        raise ValueError('Please check publication date')

    return df, date_pub

def _get_data_uk_countries_ons():
    """Data for UK countries based on PCR (SGTF, N-gege, ...)

    Shifted 14 days to estimated date of onset. There doesn't seem to be
    a background level (frequencies are too high for false detection to matter).

    Sampled 1x per week.

    # https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/conditionsanddiseases/bulletins/coronaviruscovid19infectionsurveypilot/29january2021#positive-tests-that-are-compatible-with-the-new-uk-variant
    # https://www.ons.gov.uk/visualisations/dvc1163/countrybyvar/datadownload.xlsx
    """
    df, pub_date = load_uk_ons_gov_country_by_var()

    c_names = ['England', 'Wales', 'Northern Ireland', 'Scotland']
    c_names = {cn: cn for cn in c_names}
    c_names['Northern Ireland'] = 'N. Ireland'

    shifted_dates = df.index - pd.Timedelta(14, 'd')

    cdict = {}

    # combined data for entire UK
    df_uk = pd.DataFrame(index=shifted_dates, data=dict(nnew=0., noth=0.))

    for cn in c_names:

        pnew = np.array(df[f'{cn}:pnew'], dtype=float)
        poth = np.array(df[f'{cn}:poth'], dtype=float)

        cdf = pd.DataFrame(
            dict(f_b117=pnew/(pnew + poth), or_b117=pnew/poth),
            index=shifted_dates
            )

        population = uk_countries_pop[cn]
        df_uk['nnew'] += population / 100 * pnew
        df_uk['noth'] += population / 100 * poth

        # resample, make sure to include point n-4.
        # (don't trust the last few points)
        n = len(cdf)
        cdf = cdf.iloc[(n-3)%7::7]
        cdict[f'{c_names[cn]} (SGTF; {pub_date})'] = cdf


    df_uk = df_uk.iloc[(len(df_uk)-3)%7::7]
    df_uk['f_b117'] = df_uk['nnew']/(df_uk['noth'] + df_uk['nnew'])
    df_uk['or_b117'] = df_uk['nnew']/df_uk['noth']

    cdict[f'UK (SGTF; {pub_date})'] = df_uk

    return cdict


def get_ckeys_metadata(cdict):
    """Return dataframe with metadata for per-country data.

    Parameters:

    - cdict: dict (or set, list) with keys such as
      'NL (seq; 2021-01-20)'.

    Return:

    - DataFrame with columns:
     key, date, ccode, uk_part, en_part, is_seq, is_sgtf.

    """
    ## Typical keys can look like this:
    # 'England (SGTF; multiple regions)',
    # 'East Midlands (SGTF)',
    # 'England (SGTF; 2021-01-29)',
    # 'N. Ireland (SGTF; 2021-01-29)',
    # 'Scotland (SGTF; 2021-01-29)',
    # 'South East England (seq, 2...',
    # 'UK (SGTF; 2021-01-29)',
    # 'DK (seq; 26 jan)',
    # 'NL OMT-advies #96',
    # 'NL (seq; 2021-01-20)',
    # 'PT (seq; 19 jan)',
    # 'CH (seq; 26 jan)'

    # Store dict keys into dataframe.
    records = []
    for key in cdict.keys():
        m = re.search('202.-..-..', key)
        if m:
            date = m.group(0)
        else:
            date = None

        m = re.match('([A-Z][A-Z])', key)
        if m:
            ccode = m.group(1)
        else:
            ccode = None

        m = re.match('(England|Wales|N. Ireland|Scotland) \(SGTF; 202', key)
        if m:
            uk_part = m.group(1)
        else:
            uk_part = None

        if ccode or uk_part:
            en_part = None
        else:
            m1 = re.match('(.*) \(SGTF', key)
            m2 = re.match('(South East England) \(seq', key)
            if m1:
                date = '2021-01-19'
                en_part = m1.group(1)
            elif m2:
                en_part = m2.group(1)
            else:
                raise ValueError(f'key={key}: how to categorize?')


        is_seq = 'seq' in key
        is_sgtf = 'SGTF' in key

        records.append(dict(
            key=key, date=date, ccode=ccode, uk_part=uk_part,
            en_part=en_part, is_seq=is_seq, is_sgtf=is_sgtf
            ))

    df = pd.DataFrame.from_records(records).set_index('key')
    return df


def filter_countries(cdict, select):
    """Select countries to display.

    - cdict: dict, key=region, value=dataframe.
    - select: selection preset str or comma-separated str.


        - all: everything (a lot)
        - all_recent: almost everything
        - picked: hand-picked selection.
        - countries: full countries; UK counts as one.
        - countries_uk: full countries and parts of UK
        - uk_parts: UK countries
        - eng_parts: England parts.
        - None: equivalent to 'picked'.
        - NL_DK_SEE_20210119: data from NL, DK, SEE available on 19 Jan
        - DK_SEE_20210101: data from DK, SEE available on 1 Jan
    """

    df = get_ckeys_metadata(cdict)

    # build a couple of selections (list of keys)
    countries = []
    for ccode in df.loc[~df['ccode'].isna(), 'ccode'].unique():
        countries.append(df.loc[df['ccode'] == ccode].iloc[-1].name)

    uk_parts = list(df.loc[~df['uk_part'].isna()].index)
    eng_parts = [x
                 for x in df.loc[~df['en_part'].isna()].index
                 if not 'seq, 2020-12-21' in x # older sequencing data
                 ]

    # all except duplicates for the same region.
    regions_seen = {} # key: (ccode, uk_part, en_part), value=(date, key)
    for i in range(len(df)):
        tup = tuple(df.iloc[i][['ccode', 'uk_part', 'en_part']])
        regions_seen[tup] = (df.iloc[i]['date'], df.index[i])
    all_recent = [k for _date, k in regions_seen.values()]

    keys = []
    if select is None:
        select = 'picked'
    for sel in 'picked' if select is None else select.split(','):
        if sel == 'countries':
            keys += countries
        elif sel == 'countries_uk':
            keys += (countries + uk_parts)
        elif sel == 'uk_parts':
            keys += uk_parts
        elif sel == 'eng_parts':
            keys += eng_parts
        elif sel == 'NL_DK_SEE_20210119':
            keys += ['DK (seq; 2021-01-01)', 'NL (seq; 2021-01-19; OMT)',
                     'South East England (seq, 2020-12-21)', 'South East England (seq, 2020-12-31)',
                     ]
        elif sel == 'DK_SEE_20210101':
            keys += ['DK (seq; 2021-01-01)',
                     'South East England (seq, 2020-12-21)', 'South East England (seq, 2020-12-31)',
                     ]
        elif sel == 'all':
            keys = list(df.index)
        elif sel == 'all_recent':
            keys += all_recent
        elif sel == 'picked':
            keys += countries + uk_parts + [
                'England (SGTF; multiple regions; 2021-01-15)', 'South East England (seq, 2020-12-31)'
                ]


        else:
            raise KeyError(f'selection {sel!r}')

    return {k:cdict[k] for k in keys}

def get_data_countries(select=None, subtract_eng_bg=True):
    """Return dict, key=description, value=dataframe with Date, n_pos, f_b117, or_b117.

    - subtract_eng_bg: whether to subtract background values for England regions.
    - select: optional selection.
    """

    cdict_eng = _get_data_England_regions(subtract_bg=subtract_eng_bg)

    cdict = {
        **_get_data_countries_weeknos(),
        **cdict_eng,
        **_get_data_uk_genomic(), # this is now covered by cdict_uk
        **_get_data_uk_countries_ons(),
        }

    return filter_countries(cdict, select)

