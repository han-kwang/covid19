#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 10:23:19 2021

author: hk_nien @ twitter
"""
import nlcovidstats as nlcs
nlcs.init_data()
import numpy as np

import pandas as pd

# https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv
df=  pd.read_csv('data/COVID-19_rioolwaterdata.csv', sep=';')
print(f'Rioolwater latest date: {df["Date_measurement"].max()}')
df_saved = df.copy()
#%%

df = df_saved.copy()
df_regions = pd.read_csv('data/rioolwater_gebieden.csv', comment='#')

# df will now include columns: Date_measurement, RNA_flow_per_100000, Inwonertal
df = df.join(
    df_regions[['RWZI_AWZI_code', 'Inwonertal']].set_index('RWZI_AWZI_code'),
    on='RWZI_AWZI_code',
    how='inner'
    )

df = df.loc[~df['RNA_flow_per_100000'].isna()]
df['RNA_flow_abs'] = df['RNA_flow_per_100000'] * df['Inwonertal'] * 1e-5
df['Date_measurement'] = pd.to_datetime(df['Date_measurement'])

rcodes = df['RWZI_AWZI_code'].unique()
all_dates = pd.date_range(
    df['Date_measurement'].min(), df['Date_measurement'].max(),
    freq='1 d'
    )

# Summed dataframe. Columns rflow_abs: total RNA flow;
# pop: population.
sum_df = pd.DataFrame(dict(rflow_abs=0.0, pop=0.0), index=all_dates)

for i, rcode in enumerate(rcodes):
    # DataFrame for one region
    r_df = pd.DataFrame(index=all_dates)
    r_df['RNA_flow_abs'] = df.query('RWZI_AWZI_code == @rcode').set_index('Date_measurement')['RNA_flow_abs']
    r_df = r_df.interpolate()
    # interpolation may cause leading NaN values before start of measurements.
    # Handle those.

    r_df['pop'] = df_regions.query('RWZI_AWZI_code == @rcode').iloc[0]['Inwonertal']
    r_df.loc[r_df['RNA_flow_abs'].isna(), 'pop'] = 0
    sum_df['rflow_abs'] += r_df['RNA_flow_abs']
    sum_df['pop'] += r_df['pop']
    print(f'\rProcessing region {i+1}/{len(rcodes)}', end='')

#%% calculate R
sum_df['rflow_per_capita'] = sum_df['rflow_abs'] / sum_df['pop']
delay = 5 # days

def get_gfac(s):
    """From series s"""
    # growth factor week-on-week
    gfac = s.shift(-3) / s.shift(3)
    return gfac.shift(-delay)

sum_df['GF'] = get_gfac(sum_df['rflow_per_capita'])
sum_df['GF_smooth'] = get_gfac(
    sum_df['rflow_per_capita'].rolling(21, win_type='hamming', center=True).sum()
    )

import matplotlib.pyplot as plt
import tools
plt.close('all')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
fig, (axf, axR) = plt.subplots(2, 1, sharex=True, tight_layout=True, figsize=(10, 6))
axf.plot(sum_df['rflow_per_capita'])
axf.set_yscale('log')
axf.set_title('SARS-CoV-2 RNA in rioolwater, per persoon, per dag.')
axf.grid(True, 'minor', 'y')
axR.set_title(f'Week-op-week groeifactor.')
axR.plot(sum_df['GF'], label=f'Rioolwater (vertraging {delay} d)', color='#aaaaaa')
axR.plot(sum_df['GF_smooth'], label=f'Rioolwater (gladgestreken)', color=colors[0],
         ls='--')
axR.plot(nlcs.DFS['Rt_rivm']['R']**(7/4), label='o.b.v RIVM R', color=colors[1])
axR.set_xlim(sum_df.index[0], sum_df.index[1-delay if delay > 2 else -1])
axR.set_ylim(0.3, 3)
axR.legend()
axR.grid()

for ax in axR, axf:
    tools.set_xaxis_dateformat(ax, xlabel='Datum monstername', maxticks=20)
fig.show()

