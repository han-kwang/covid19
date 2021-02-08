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


def _add_meta_record(reclist, desc, mdict, refs):
    """Add record to reclist; record is a dict with keys:

        desc=desc,
        **mdict,
        refs->tuple of URLs

    mdict['date'] will be converted to DateTime object.
    """
    refs = tuple(refs)

    rec = {'desc': desc,
           **mdict,
           'refs': refs,
           }

    if 'date' in rec:
        rec['date'] = pd.to_datetime(rec['date'])

    reclist.append(rec)

def _get_data_uk_genomic():

    # https://twitter.com/hk_nien/status/1344937884898488322

    # data points read from plot (as ln prevalence)
    seedata = {
        '2020-12-21': [
            dict(date='2020-12-21', is_recent=False, is_seq=True, en_part='South East England'),
            ['https://t.co/njAXPsVlvb?amp=1'],
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
            dict(date='2020-12-31', is_recent=True, is_seq=True, en_part='South East England'),
            ['https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-12-31-COVID19-Report-42-Preprint-VOC.pdf'
             ],
            ['2020-10-31', -2.1],
            ['2020-11-08', -1.35],
            ['2020-11-15', -0.75],
            ['2020-11-22', -0.05],
            ['2020-11-29', 0.05],
            ]
        }

    cdict = {}
    meta_records = []
    for report_date, records in seedata.items():
        df = pd.DataFrame.from_records(records[2:], columns=['Date', 'ln_odds'])
        df['Date'] = pd.to_datetime(df['Date'])
        odds = np.exp(df['ln_odds'])
        df['f_b117'] = odds / (1 + odds)
        df = df[['Date', 'f_b117']].set_index('Date')
        desc = f'South East England (seq, {report_date})'
        cdict[desc] = df
        _add_meta_record(meta_records, desc, records[0], records[1])

    return cdict, meta_records



def _get_data_countries_weeknos():
    """Countries with f_117 data by week number.

    Return dataframe with metadata and dict of dataframes.
    """


    # All country records are ('{year}-W{weekno}-{weekday}', fraction_b117)
    # Item 0 in each list: metadata
    # Item 1 in each list: source URLs
    country_records = {
        'DK (seq; 2021-01-01)': [
            dict(ccode='DK', date='2021-01-01', is_seq=True, is_recent=False),
            ['https://covid19.ssi.dk/-/media/cdn/files/opdaterede-data-paa-ny-engelsk-virusvariant-sarscov2-cluster-b117--01012021.pdf?la=da'],
            ('2020-W49-4', 0.002),
            ('2020-W50-4', 0.005),
            ('2020-W51-4', 0.009),
            ('2020-W52-4', 0.023)
            ],
        'DK (seq; 2021-02-05)': [
            dict(ccode='DK', date='2021-02-05', is_seq=True, is_recent=True),
            ['https://www.covid19genomics.dk/statistics'],
            ('2020-W48-4', 0.002),
            ('2020-W49-4', 0.002),
            ('2020-W50-4', 0.004),
            ('2020-W51-4', 0.008),
            ('2020-W52-4', 0.020),
            ('2020-W53-4', 0.024),
            ('2021-W01-4', 0.040), # last updated 2021-02-05
            ('2021-W02-4', 0.075),
            ('2021-W03-4', 0.128),
            ('2021-W04-4', 0.191), # last updated 2021-02-05
            ],
        'NL (seq; 2021-01-19; OMT)': [
            dict(ccode='NL', date='2021-01-01', is_seq=True, is_recent=False),
            ['https://www.tweedekamer.nl/kamerstukken/brieven_regering/detail?id=2021Z00794&did=2021D02016',
             'https://www.rivm.nl/coronavirus-covid-19/omt'],
            ('2020-W49-4', 0.011),
            ('2020-W50-4', 0.007),
            ('2020-W51-4', 0.011),
            ('2020-W52-4', 0.014),
            ('2020-W53-4', 0.052),
            ('2021-W01-4', 0.119), # preliminary / calculation error (0.135???)
            ],
        'NL (seq; 2021-02-07)': [
            dict(ccode='NL', date='2021-02-07', is_seq=True, is_recent=True),
            ['https://www.tweedekamer.nl/kamerstukken/brieven_regering/detail?id=2021Z00794&did=2021D02016',
             'https://www.tweedekamer.nl/sites/default/files/atoms/files/20210120_technische_briefing_commissie_vws_presentati_jaapvandissel_rivm_0.pdf',
             'https://www.tweedekamer.nl/downloads/document?id=00588209-3f6b-4bfd-a031-2d283129331c&title=98e%20OMT%20advies%20deel%201%20en%20kabinetsreactie',
             'https://www.tweedekamer.nl/downloads/document?id=be0cb7fc-e3fd-4a73-8964-56f154fc387e&title=Advies%20n.a.v.%2099e%20OMT%20deel%202.pdf'
             ],
            ('2020-W49-5', 0.011), # OMT #96 >>
            ('2020-W50-5', 0.007),
            ('2020-W51-5', 0.011),
            ('2020-W52-5', 0.020),
            ('2020-W53-5', 0.050), # << OMT #96
            ('2021-W01-5', 0.090), # TK briefing (read from figure ±0.005)
            ('2021-W02-5', 0.198), # OMT #98 (31 Jan)
            ('2021-W03-5', 0.241), # OMT #99
            ],
        'UK (seq; 2021-01-21)': [
            dict(ccode='UK', date='2021-01-21', is_seq=True, is_recent=True),
            ['https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-risk-related-to-spread-of-new-SARS-CoV-2-variants-EU-EEA-first-update.pdf',
            ],
            # Fig. 2. (traced, +/- 0.001 accuracy)
            ('2020-W43-4', 0.003),
            ('2020-W44-4', 0.008),
            ('2020-W45-4', 0.026),
            ('2020-W46-4', 0.063),
            ('2020-W47-4', 0.108),
            ('2020-W48-4', 0.101),
            ('2020-W49-4', 0.140),
            ('2020-W50-4', 0.333),
            ('2020-W51-4', 0.483),
            ('2020-W52-4', 0.539),
            ('2020-W53-4', 0.693),
            # ('2021-W01-4', ...),
            ],
        'PT (seq; 2021-02-01)': [
            dict(ccode='PT', date='2021-02-01', is_seq=True, is_recent=True),
            ['https://virological.org/t/tracking-sars-cov-2-voc-202012-01-lineage-b-1-1-7-dissemination-in-portugal-insights-from-nationwide-rt-pcr-spike-gene-drop-out-data/600',
             'https://virological.org/t/tracking-sars-cov-2-voc-202012-01-lineage-b-1-1-7-dissemination-in-portugal-insights-from-nationwide-rt-pcr-spike-gene-drop-out-data/600/4',
             ],
            ('2020-W49-4', 0.019),
            ('2020-W50-4', 0.009),
            ('2020-W51-4', 0.013),
            ('2020-W52-4', 0.019),
            ('2020-W53-4', 0.032),
            ('2021-W01-4', 0.074),
            ('2021-W02-4', 0.133),
            ('2021-W03-4', 0.247),
            ],
        'CH (seq; 2021-02-05)': [
            dict(ccode='CH', date='2021-02-05', is_seq=True, is_recent=True),
            ['https://sciencetaskforce.ch/nextstrain-phylogentische-analysen/'],
            ('2020-W51-4', 0.0004),
            ('2020-W52-4', 0.0043),
            ('2020-W53-4', 0.0074),
            ('2021-W01-4', 0.0153),
            ('2021-W02-4', 0.0329),
            ('2021-W03-4', 0.0881),
            ('2021-W04-4', 0.158), # last updated ca. 2021-02-05
            ],
        # https://assets.gov.ie/121054/55e77ccd-7d71-4553-90c9-5cd6cdee7420.pdf (p. 53) up to wk 1
        # https://assets.gov.ie/121662/184e8d00-9080-44aa-af74-dbb13b0dcd34.pdf (p. 2, bullet 8) wk 2/3
        'IE (SGTF; 2021-01-28)': [
            dict(ccode='IE', date='2021-01-28', is_seq=False, is_sgtf=True, is_recent=True),
            ['https://assets.gov.ie/121054/55e77ccd-7d71-4553-90c9-5cd6cdee7420.pdf', # (p. 53) up to wk 1
             'https://assets.gov.ie/121662/184e8d00-9080-44aa-af74-dbb13b0dcd34.pdf', # (p. 2, bullet 8) wk 2/3
             ],
            ('2020-W50-4', 0.014),
            ('2020-W51-4', 0.086),
            ('2020-W52-4', 0.163),
            ('2020-W53-4', 0.262),
            ('2021-W01-4', 0.463), # 21 Jan
            ('2021-W02-4', 0.58),
            ('2021-W03-4', 0.63), # 28 Jan
            ]
        }

    cdict = {}
    meta_records = []
    for desc, records in country_records.items():
        cdict[desc] = _convert_ywd_records(records[2:], ['f_b117'])
        _add_meta_record(meta_records, desc, records[0], records[1])

    return cdict, meta_records


#%%


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

    pub_date = '2021-01-15'
    cdict = {f'England (SGTF; multiple regions; {pub_date})': df_combi}
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

        cdict[f'{region} (SGTF; {pub_date}'] = df

    # convert to estimated new cases per day.
    for key, df in cdict.items():

        region = re.match(r'(.*) \(.*', key).group(1)
        if region == 'England':
            region = f'England (multiple regions; {pub_date})'

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

    meta_records = []
    for desc in cdict.keys():
        region = re.match('(.*) \(', desc).group(1)
        record = dict(
            desc=desc,
            date=pd.to_datetime(pub_date),
            en_part=region,
            is_recent=True,
            is_seq=False,
            is_sgtf=True,
            refs=('https://doi.org/10.1101/2021.01.13.21249721',)
            )
        meta_records.append(record)

    return cdict, meta_records



def load_uk_ons_gov_country_by_var():
    """Get data based on data/ons_gov_uk_country_by_var.xlsx.

    Return:

    - dataframe
    - date_pub
    - tuple of source URLs

    Dataframe layout:

    - index: Date.
    - columns: {country_name}:{suffix}

    with suffix = 'pnew', 'pnew_lo', 'pnew_hi', 'poth', ..., 'pnid', ...

    for percentages new UK variant, CI low, CI high,
    other variant, not-identified.
    """

    refs = [
        'https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/conditionsanddiseases/bulletins/coronaviruscovid19infectionsurveypilot/29january2021#positive-tests-that-are-compatible-with-the-new-uk-variant',
        'https://www.ons.gov.uk/visualisations/dvc1163/countrybyvar/datadownload.xlsx',
        ]

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

    return df, date_pub, refs

def _get_data_uk_countries_ons():
    """Data for UK countries based on PCR (SGTF, N-gege, ...)

    Shifted 14 days to estimated date of onset. There doesn't seem to be
    a background level (frequencies are too high for false detection to matter).

    Sampled 1x per week.

    # https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/conditionsanddiseases/bulletins/coronaviruscovid19infectionsurveypilot/29january2021#positive-tests-that-are-compatible-with-the-new-uk-variant
    # https://www.ons.gov.uk/visualisations/dvc1163/countrybyvar/datadownload.xlsx
    """
    df, pub_date, refs = load_uk_ons_gov_country_by_var()

    c_names = ['England', 'Wales', 'Northern Ireland', 'Scotland']
    c_names = {cn: cn for cn in c_names}
    c_names['Northern Ireland'] = 'N. Ireland'

    shifted_dates = df.index - pd.Timedelta(14, 'd')

    cdict = {}
    meta_records = []

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
        desc = f'{c_names[cn]} (SGTF; {pub_date})'
        cdict[desc] = cdf

        meta_records.append(dict(
            desc=desc,
            date=pd.to_datetime(pub_date),
            uk_part=cn,
            is_seq=False,
            is_sgtf=True,
            refs=refs,
            ))


    df_uk = df_uk.iloc[(len(df_uk)-3)%7::7]
    df_uk['f_b117'] = df_uk['nnew']/(df_uk['noth'] + df_uk['nnew'])
    df_uk['or_b117'] = df_uk['nnew']/df_uk['noth']

    cdict[f'UK (SGTF; {pub_date})'] = df_uk

    return cdict, meta_records


def _get_data_ch_parts():
    """Note: this is daily data, not weekly data"""

    region_records = {
        'Genève (2021-02-07)': [
            dict(ch_part='Genève', date='2021-02-07', is_recent=True, is_pcr=True),
            ['https://ispmbern.github.io/covid-19/variants/'],
            ('2021-01-13', 0.1817),
            ('2021-01-14', 0.09823),
            ('2021-01-15', 0.1932),
            ('2021-01-16', 0.2441),
            ('2021-01-17', 0.2124),
            ('2021-01-18', 0.2499),
            ('2021-01-19', 0.2167),
            ('2021-01-20', 0.1903),
            ('2021-01-21', 0.1661),
            ('2021-01-22', 0.2907),
            ('2021-01-23', 0.2557),
            ('2021-01-24', 0.3348),
            ('2021-01-25', 0.2665),
            ('2021-01-26', 0.4243),
            ('2021-01-27', 0.4792),
            ('2021-01-28', 0.4893),
            ('2021-01-29', 0.5135),
            ('2021-01-30', 0.558),
            ('2021-01-31', 0.5749),
            ],
        'Zürich (2021-02-07)': [
            dict(ch_part='Zürich', is_recent=True, rebin=3),
            ['https://ispmbern.github.io/covid-19/variants/'],
            ('2021-01-06', 0.0007223),
            ('2021-01-07', 0.03684),
            ('2021-01-08', 0.01697),
            ('2021-01-09', -0.0003611),
            ('2021-01-10', 0.04912),
            ('2021-01-11', 0.02564),
            ('2021-01-12', -0.0003611),
            ('2021-01-13', 0.02961),
            ('2021-01-14', 0.1116),
            ('2021-01-15', 0.1434),
            ('2021-01-16', 0.0003611),
            ('2021-01-17', 0.08451),
            ('2021-01-18', -0.0007223),
            ('2021-01-19', 0.1492),
            ('2021-01-20', 0.06284),
            ('2021-01-21', 0.277),
            ('2021-01-22', 0.05273),
            ('2021-01-23', 0.2232),
            ('2021-01-24', 0.1672),
            ('2021-01-25', 0.2004),
            ('2021-01-26', 0.1192),
            ('2021-01-27', 0.2867),
            ('2021-01-28', 0.1571),
            ('2021-01-29', 0.08234),
            ('2021-01-30', 0.2867),
            ('2021-01-31', 0.2777),
            ('2021-02-01', 0.2929),
            ('2021-02-02', 0.1495),
            ('2021-02-03', -0.0003611),
        ]
        }

    cdict = {}
    meta_records = []
    for desc, records in region_records.items():
        df = pd.DataFrame.from_records(records[2:], columns=['sample_date', 'f_b117'])
        df['sample_date'] = pd.to_datetime(df['sample_date'])
        df = df.set_index('sample_date')
        rebin = records[0]['rebin'] if 'rebin' in records[0] else None
        if rebin:
            # rebin noisy dataset
            df = df.iloc[-(len(df) // rebin * rebin):]
            df = df.rolling(window=rebin, center=True).mean().iloc[rebin//2::rebin]
        cdict[desc] = df


        _add_meta_record(meta_records, desc, records[0], records[1])

    return cdict, meta_records


def filter_countries(cdict, meta_df, select):
    """Select countries to display.

    - cdict: dict, key=region, value=dataframe.
    - meta_df: dataframe with metadata (columns ccode, is_recent, etc.).
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

    Return:

    - cdict: selected cdict
    - meta_df: subset of original meta_df dataframe
    """
    df = meta_df

    # build a couple of selections (list of keys)
    countries = []
    countries = list(df.index[~df['ccode'].isna() & df['is_recent']])
    uk_national = list(df.index[(df['ccode']=='UK')])
    uk_parts = list(df.index[~df['uk_part'].isna()])
    ch_parts = list(df.index[~df['ch_part'].isna() | (df['ccode'] == 'CH')])
    eng_parts = list(df.index[~df['en_part'].isna() & df['is_recent']])
    all_recent = list(df.index[df['is_recent']])

    keys = []
    if select is None:
        select = 'picked'
    for sel in 'picked' if select is None else select.split(','):
        if sel == 'countries':
            keys += countries
        elif sel == 'countries_uk':
            keys += (countries + uk_parts)
        elif sel == 'uk':
            keys += uk_national + uk_parts + eng_parts
        elif sel == 'ch':
            keys += ch_parts
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

    keys = df.loc[keys].index.unique()
    new_cdict = {k:cdict[k] for k in keys}
    new_mdf = df.loc[keys]

    return new_cdict, new_mdf



def get_data_countries(select=None, subtract_eng_bg=True):
    """Return dict, key=description, value=dataframe with Date, n_pos, f_b117, or_b117.

    - subtract_eng_bg: whether to subtract background values for England regions.
    - select: optional selection.
    """

    cdicts_and_records = [
        # each entry is a tuple (cdict, meta_records)
        _get_data_countries_weeknos(),
        _get_data_uk_genomic(),
        _get_data_uk_countries_ons(),
        _get_data_England_regions(subtract_bg=subtract_eng_bg),
        _get_data_ch_parts(),
        ]

    cdict = {}
    meta_records = [] # list of dicts
    for cd, rs in cdicts_and_records:
        cdict.update(cd)
        meta_records.extend(rs)

    meta_df = pd.DataFrame.from_records(meta_records).set_index('desc')

    # refs column should be the final column
    columns = ['ccode', 'date', 'is_seq', 'is_sgtf', 'is_pcr', 'is_recent', 'en_part', 'uk_part', 'refs']
    columns.extend(set(meta_df.columns).difference(columns)) # just in case

    meta_df = meta_df[columns]

    # cleanup columns (NaN values etc).
    for c in list(meta_df.columns):
        if c.startswith('is_'):
            meta_df[c] = (meta_df[c] == True)
        elif meta_df[c].dtype == 'O':
            meta_df.loc[meta_df[c].isna(), c] = None

    return filter_countries(cdict, meta_df, select)


if __name__ == '__main__':
    # just for testing
    cdict, mdf = get_data_countries('picked')
    cdict, mdf = get_data_countries('ch')
