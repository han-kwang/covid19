#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 12:34:25 2021

@author: hk_nien @ Twitter
"""



import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nlcovidstats as nlcs
from tools import set_xaxis_dateformat

def get_variant_dataframe_fromcsv():
    """Return DataFrame with variant counts as columns. Crap. CSV outdated."""
    # https://data.rivm.nl/covid-19/COVID-19_varianten.csv
    fname = 'data/COVID-19_varianten.csv'

    df = pd.read_csv(fname, sep=';')
    df['Variant'] = df['Variant_name'] + '/' + df['Variant_code']
    df.drop(columns=['Variant_code', 'Variant_name'], inplace=True)
    df.set_index('Date_of_statistics_week_start', inplace=True)
    ndf = pd.DataFrame(index=pd.to_datetime(sorted(df.index.unique())))

    for vcode in df['Variant'].unique():
        df_sel = df.loc[df['Variant'] == vcode]
        if len(ndf.columns) == 0:
            # First.
            ndf['Sample_size'] = df_sel['Sample_size']
        ndf[vcode] = df_sel['Variant_cases']

    # Change index to mid-week (Thu) date.
    ndf['Date_midweek'] = ndf.index - pd.Timedelta('3.5 d')
    ndf.set_index('Date_midweek', inplace=True)

    isodates = [
        f'{x.isocalendar()[0]}-W{x.isocalendar()[1]:02d}'
        for x in ndf.index
        ]
    ndf.insert(0, 'Week_no', isodates)

    return ndf

def get_variant_dataframe_from_html_table():

    df = pd.read_csv('data/covid-19_variants_nl.csv', sep='\t', comment='#')
    columns = [c for c in df.columns if re.match(r'(Weeknummer|Totaal|..../..) ?$', c)]
    df = df[columns]
    df.rename(columns={'Weeknummer ': 'Weeknummer'}, inplace=True)
    df.set_index('Weeknummer', inplace=True)
    df.index.name = 'Variant'
    df.columns.name = 'Weeknummer'
    df = df.T

    var_map = {
      'Alfa* (B.1.1.7) ': 'Alfa',
      'Alfa met E484K mutatie ': 'Alfa+E484K',
      'Beta (B.1.351)  ': 'Beta',
      'Gamma (P.1)  ': 'Gamma',
      'Delta (B.1.617.2) ': 'Delta',
      'Eta (B.1.525) ': 'Eta',
      'Epsilon (B.1.427/4.29) ': 'Epsilon',
      'Theta (P.3) ': 'Theta',
      'Kappa (B.1.617.1) ': 'Kappa',
      'Variant Bretagne (B.1.616) ': 'B.1.616',
      'B.1.620 ': 'B.1.620',
      'Colombiaanse variant (B.1.621) ': 'B.1.621',
      'Lambda (C.37) ': 'Lambda',
      'Iota (B.1.526) ': 'Iota',
      'Zeta (P.2) ': 'Zeta',
     }
    df.rename(columns=var_map, inplace=True)
    df.drop(columns='Alfa+E484K', inplace=True)  # already counted in Alfa.
    df.drop(index='Totaal ', inplace=True)
    # Create new datetime index
    df.reset_index(inplace=True)


    import datetime
    dates = [
        datetime.datetime.strptime(f'{d}/4', '%Y/%W /%w')
        for d  in df['Weeknummer']
        ]
    weeknos = [
        f'{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}'
        for d in dates
        ]
    df['Date_sample'] = dates
    df['Weekno'] = weeknos
    df = df.set_index('Date_sample').drop(columns='Weeknummer')
    df = df.rename(columns={'Aantal onderzochte monsters ': 'n_samp'})
    df = df.loc[df.index.sort_values()]

    return df

def add_npos_to_df(vdf):
    """Add 'npos' column to DataFrame based on 'Weekno' column; return new DataFrame."""

    # Get positive tests NL
    nlcs.init_data(autoupdate=True)
    tdf, npop = nlcs.get_region_data('Nederland')
    tdf = pd.DataFrame(dict(
        npos=tdf['Delta'].values * npop,
        date_rep=tdf.index)
        )

    # Weeknos assumed 1 day before publication.
    tdf['Weekno'] = [
        f'{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}'
        for d in tdf['date_rep'] - pd.Timedelta('1 d')
        ]
    twdf = tdf.groupby('Weekno').sum() # DataFrame with Weekno as index, npos column.
    return vdf.join(twdf, on='Weekno', how='left')

def get_R_variant(vdf, vname='Alfa', Tg=4.0):
    """Return DataFrame with R estimate and error margin for specified variant."""

    df = vdf[['n_samp', 'Weekno', 'npos']].copy()
    df['nv_samp'] = vdf[vname]
    df['nvar'] = np.around(df['nv_samp'] * df['npos'] / df['n_samp'], 1)
    idxs = df.index
    gfacs = np.around(df.loc[idxs[1:], 'nvar'].values / df.loc[idxs[:-1], 'nvar'].values, 2)

    # calculus with relative errors = sqrt(relative variance)
    # g = a/b, a +/- re*a, b +/- re*b; g +/- g*re_c

    rva = 1/df.loc[idxs[1:], 'nv_samp'].values
    rvb = 1/df.loc[idxs[:-1], 'nv_samp'].values
    rvc = rva + rvb
    gfac_sigmas = np.around(gfacs * np.sqrt(rvc), 2)
    deltats = idxs[1:] - idxs[:-1]
    gdates = idxs[:-1] + 0.5*deltats + pd.Timedelta(12, 'h')
    expons = np.array(Tg/deltats.days, dtype=float)
    Rs = gfacs ** expons
    Rsigmas = Rs * expons * gfac_sigmas / gfacs
    Rs[gfac_sigmas > 0.8*gfacs] = np.nan
    # log(g+/-d) = log(g) +/- d/g


    gdf = pd.DataFrame(index=gdates, data=dict(
        gfac=gfacs,
        gfac_sigma=gfac_sigmas,
        date_R=gdates - pd.Timedelta(3, 'd'),
        R=Rs,
        R_sigma=Rsigmas
        ))

    return gdf

if 1:  # set to 0 for debugging
    vdf = get_variant_dataframe_from_html_table()
    vdf = add_npos_to_df(vdf)
else:
    vdf_store = vdf.copy()
    df = vdf_store.copy()
    pd.options.display.width=100


plt.close('all')
fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)

for vname in ['Alfa', 'Gamma', 'Delta']:
    rdf = get_R_variant(vdf, vname)
    # Remove bad data
    rdf.loc[rdf['R_sigma'] > 1, 'R'] = np.nan
    ax.errorbar(rdf['date_R'], rdf['R'], rdf['R_sigma'], label=vname,
                capsize=3
                )
    ax.scatter(rdf['date_R'], rdf['R'])

# ax.set_ylim(0.5, 2)
ax.set_ylabel('R')
ax.set_title('Reproductiegetal per variant (Nederland)\n'
             '(Aanname: generatie-interval 4 dagen)')
ax.legend(loc='upper left')
nlcs._add_restriction_labels(ax, rdf['date_R'][0], rdf['date_R'][-1], flagmatch='RGraph')




set_xaxis_dateformat(ax)
fig.show()



